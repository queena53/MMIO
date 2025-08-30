import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
import nibabel as nib

from PIL import Image
from collections import Counter, defaultdict

def save_as_nifti(data, save_path):
    """
    將數據保存為 NIfTI 格式
    :param data: 3D numpy array, 數據形狀應為 [D, H, W]
    :param save_path: 保存的文件路徑
    """
    nifti_img = nib.Nifti1Image(data, affine=np.eye(4))  # 使用單位矩陣作為仿射矩陣
    nib.save(nifti_img, save_path)
    print(f"✅ Saved NIfTI file to {save_path}")

def save_slice_40_as_png(dataset, save_root, prefix):
    os.makedirs(save_root, exist_ok=True)
    
    for i in range(len(dataset)):
        # 解包三個值
        x, y, file_path = dataset[i]
        
        if not file_path.endswith("slice_40.npy"):
            continue  # ⛔ 跳過非 slice_40 的檔案

        x = x.squeeze(0).numpy()  # [128, 128, 128]
        # 保存整個 3D 體積數據為 NIfTI 文件
        file_name = os.path.basename(file_path).replace(".npy", ".nii")
        save_path = os.path.join(save_root, f"{prefix}_{file_name}")
        save_as_nifti(x, save_path)

        # 取得第 64 張 axial slice（中間切片）
        z = 64
        slice_img = x[z, :, :]
        norm_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
        slice_img_rgb = (np.stack([norm_slice] * 3, axis=-1) * 255).astype(np.uint8)
        img = Image.fromarray(slice_img_rgb)

        file_name = os.path.basename(file_path).replace(".npy", ".png")
        save_path = os.path.join(save_root, f"{prefix}_{file_name}")
        img.save(save_path)

        print(f"✅ Saved PNG for {file_path} → {save_path}")

class NPY3DDataset(Dataset):
    def __init__(self, file_paths, labels, target_shape=(1, 128, 128, 128)):
        self.file_paths = file_paths
        self.labels = labels
        self.target_shape = target_shape  # 目标形状

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        x = np.load(self.file_paths[idx])
        # print(f"Loaded .npy shape: {x.shape}")

        #load binary mask *x


        # ⛔ 保證輸入是4D
        if x.ndim != 4:
            raise ValueError(f"Input shape should be (1, D, H, W), got {x.shape}")

        x = self.pad_to_target_shape(x, self.target_shape)  # padding 成 (1, 128, 128, 128)
        x = torch.from_numpy(x).float()

        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # print(f"📂 File: {self.file_paths[idx]}, Label: {y.item()}")
        file_path = os.path.join(*self.file_paths[idx].split(os.sep)[-2:])
        # print(f"📂 File: {file_path}, Label: {y.item()}")

        return x, y, file_path
    
    def pad_to_target_shape(self, x, target_shape):
        _, d, h, w = x.shape
        target_c, target_d, target_h, target_w = target_shape

        pad_d = target_d - d
        pad_h = target_h - h
        pad_w = target_w - w

        pad_d_front = pad_d // 2
        pad_d_back = pad_d - pad_d_front

        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top

        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        x = np.pad(x,
                   pad_width=((0, 0), (pad_d_front, pad_d_back), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                   mode='constant', constant_values=0)
        return x

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
                if slice_file.endswith(".npy") and "slice_" in slice_file:
                    slice_number = int(slice_file.split("_")[1].split(".")[0])
                    if 20 <= slice_number <= 65:

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

def get_folds(config):
    data_root = config["data_root"]
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
            
            val_fold_idx = (fold_idx + 1) % num_folds
            val_patients = set(fold_dict[str(val_fold_idx)])

            train_patients = set()
            for k, v in fold_dict.items():
                if int(k) != fold_idx and int(k) != val_fold_idx:
                    train_patients.update(v)

            train_files, train_labels = [], []
            val_files, val_labels = [], []

            for patient in train_patients:
                info = patient_to_info[patient]
                print(f"Train patient: {patient}, diagnosis: {info['diagnosis']}")

                label = diagnosis_to_label[info["diagnosis"]]
                train_files.extend(info["paths"])
                train_labels.extend([label] * len(info["paths"]))
                    

            for patient in val_patients:
                info = patient_to_info[patient]
                print(f"Val patient: {patient}, diagnosis: {info['diagnosis']}")

             
             
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

