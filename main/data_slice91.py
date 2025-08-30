import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
from PIL import Image

from collections import Counter, defaultdict

def save_slice_40_as_png(dataset, save_root, prefix):
    os.makedirs(save_root, exist_ok=True)
    
    for i in range(len(dataset)):
        file_path = dataset.file_paths[i]
        if not file_path.endswith("slice_40.npy"):
            continue  # ⛔ 跳過非 slice_40 的檔案

        x, y = dataset[i]  # x: [1, 128, 128, 128]

        x = x[0].numpy() # [128, 128, 128]
        print(f"Loaded .npy shape: {x.shape}")  

        # 取得第 64 張 axial slice（中間切片）
        z = 64
        slice_img = x[z, :, :]
        norm_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
        slice_img_rgb = (np.stack([norm_slice]*3, axis=-1) * 255).astype(np.uint8)
        img = Image.fromarray(slice_img_rgb)

        file_name = os.path.basename(file_path).replace(".npy", ".png")
        save_path = os.path.join(save_root, f"{prefix}_{file_name}")
        img.save(save_path)

        print(f"✅ Saved PNG for {file_path} → {save_path}")

class NPY3DDataset(Dataset):
    def __init__(self, file_paths, labels, target_shape=(10, 128, 128, 128)):
        self.file_paths = file_paths
        self.labels = labels
        self.target_shape = target_shape  # 目标形状

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        x = np.load(self.file_paths[idx])
        # print(f"Loaded .npy shape: {x.shape}")

        #load binary mask *x

        # print(f"Loaded .npy shape: {x.shape}")
        x=np.pad(x, ((0, 0), (63, 62), (41, 41), (10, 9)), mode='constant', constant_values=0)

        # print(f"After padding shape: {x.shape}")    

        x= torch.from_numpy(x).float()  # shape: [1, 128, 128, 128]
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # print(f"📂 File: {self.file_paths[idx]}, Label: {y.item()}")
        return x, y

def load_subject_info(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def load_data_paths(data_root, subject_info, selected_classes):
    data = []
    for cls in selected_classes:
        cls_path = os.path.join(data_root, cls)
        if not os.path.exists(cls_path):
            continue
        for patient in os.listdir(cls_path):
            patient_dir = os.path.join(cls_path, patient)
            if patient not in subject_info:
                continue  # 跳過不在 JSON 裡的病人
            for slice_file in os.listdir(patient_dir):
                if slice_file.endswith(".npy"):
                    path = os.path.join(patient_dir, slice_file)
                    diagnosis = subject_info[patient]["Diagnosis"]
                    instrument = subject_info[patient]["Instrument"]
                    data.append({
                        "path": path,
                        "patient": patient,
                        "diagnosis": diagnosis,
                        "instrument": instrument,
                    })
    return data

def get_folds_slice91(config):
    data_root = "./noich_right_hemisphere_mask_frame22_24_slice10_class_npydata"
    selected_classes = config["selected_classes"]
    num_folds = config["num_folds"]
    subject_json = config["subject_json"]
    fixed_fold_json = config.get("5fold_json", None)


    subject_info = load_subject_info(subject_json)
    data_entries = load_data_paths(data_root, subject_info, selected_classes)

    # 組成病人資料
    patient_to_info = defaultdict(lambda: {"paths": [], "diagnosis": None, "instrument": None})
    for entry in data_entries:
        patient = entry["patient"]
        patient_to_info[patient]["paths"].append(entry["path"])
        patient_to_info[patient]["diagnosis"] = entry["diagnosis"]
        patient_to_info[patient]["instrument"] = entry["instrument"]

    diagnosis_set = sorted(set([info["diagnosis"] for info in patient_to_info.values()]))
    # diagnosis_to_label = {d: i for i, d in enumerate(diagnosis_set)}
    diagnosis_to_label = {'AD': 0, 'CAA_ICH': 1, 'CAA_CI': 2}
    folds = []

    if fixed_fold_json:
        with open(fixed_fold_json, "r") as f:
            fold_dict = json.load(f)

        all_patients = set(patient_to_info.keys())

        for fold_idx in range(num_folds):
            val_patients = set(fold_dict[str(fold_idx)])
            train_patients = all_patients - val_patients

            train_files, train_labels = [], []
            val_files, val_labels = [], []

            for patient in train_patients:
                info = patient_to_info[patient]
                label = diagnosis_to_label[info["diagnosis"]]
                train_files.extend(info["paths"])
                train_labels.extend([label] * len(info["paths"]))
                    

            for patient in val_patients:
                info = patient_to_info[patient]
                label = diagnosis_to_label[info["diagnosis"]]
                val_files.extend(info["paths"])
                val_labels.extend([label] * len(info["paths"]))

            # print(f"\nFold {fold_idx + 1}:")
            # print(f"Train Patients: {train_patients}")

            # print(f"Val Patients: {val_patients}")

            folds.append({
                "train_dataset": NPY3DDataset(train_files, train_labels),
                "val_dataset": NPY3DDataset(val_files, val_labels),
                "train_info": {p: patient_to_info[p] for p in train_patients},
                "val_info": {p: patient_to_info[p] for p in val_patients},
            })

            #儲存輸入資料的 slice_40
            for i, fold in enumerate(folds):
                print(f"\n🧪 Saving slice_40 only for Fold {i}")
                save_slice_40_as_png(fold["train_dataset"], f"fold_{i}_train_png", prefix="train")
                save_slice_40_as_png(fold["val_dataset"], f"fold_{i}_val_png", prefix="val")

    return folds

