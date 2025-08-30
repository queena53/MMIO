import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import defaultdict, Counter  # ✅ 加入 Counter
from model_medicalNet import resnet10, resnet18, resnet101
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pandas as pd

def build_config_by_test_path(test_data_path):
    base_name = os.path.basename(test_data_path)
    suffix = base_name.replace("noich_right_hemisphere_", "").replace("_class_npydata", "")
    print(f"\U0001F527 Building config for test data path: {test_data_path} with suffix: {suffix}")

    model_dir = f"./output/classification_newfold_AD_CAA_{suffix}/desnet201_ne"
    output_dir = "test_all_result"

    model_paths = [
        os.path.join(model_dir, f"desnet201_newfold_{i}_model.pth") for i in range(5)
    ]
    predict_output_paths = [
        os.path.join(output_dir, f"test_{suffix}_fold_{i}.txt") for i in range(5)
    ]

    return {
        "test_data_path": test_data_path,
        "selected_classes": ["AD", "CAA_ICH"],
        "subject_json": "subjects_data_with_abeta.json",
        "model_paths": model_paths,
        "suffix": suffix,
        "predict_output_paths": predict_output_paths
    }


# === 多個 test 資料夾統一設定 ===

test_data_paths = [
    "./SUV/noich_right_hemisphere_mask_frame5_24_slice1_class_npydata",
    "./SUV/noich_right_hemisphere_mask_frame5_12_slice1_class_npydata",
    "./SUV/noich_right_hemisphere_mask_frame22_24_slice1_class_npydata",
    "./SUVr_cerebellum/noich_right_hemisphere_suvr_cerebellum_all_mask_frame5_24_slice1_class_npydata",
    "./SUVr_cerebellum/noich_right_hemisphere_suvr_cerebellum_all_mask_frame5_12_slice1_class_npydata",
    "./SUVr_cerebellum/noich_right_hemisphere_suvr_cerebellum_all_mask_frame22_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_frontal_mask_frame5_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_frontal_mask_frame5_12_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_frontal_mask_frame22_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_occipital_mask_frame5_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_occipital_mask_frame5_12_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_occipital_mask_frame22_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_parietal_mask_frame5_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_parietal_mask_frame5_12_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_parietal_mask_frame22_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_temporal_mask_frame5_24_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_temporal_mask_frame5_12_slice1_class_npydata",
    "./SUVr_cerebellum_classified/noich_right_hemisphere_suvr_cerebellum_temporal_mask_frame22_24_slice1_class_npydata",
    "./SUVr_pons/noich_right_hemisphere_suvr_pons_all_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons/noich_right_hemisphere_suvr_pons_all_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons/noich_right_hemisphere_suvr_pons_all_mask_frame22_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_frontal_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_frontal_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_frontal_mask_frame22_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_occipital_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_occipital_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_occipital_mask_frame22_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_parietal_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_parietal_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_parietal_mask_frame22_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_temporal_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_temporal_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_temporal_mask_frame22_24_slice1_class_npydata"
]

ALL_CONFIGS = [build_config_by_test_path(path) for path in test_data_paths]


class PredictDataset(Dataset):
    def __init__(self, root_path, selected_classes, label_dict_path=None, patient_filter=None):
        self.file_list = []
        self.label_dict = {}

        for cls in selected_classes:
            cls_path = os.path.join(root_path, cls)
            if not os.path.isdir(cls_path):
                print(f"⚠️ 类别文件夹不存在: {cls_path}")
                continue

            for patient in sorted(os.listdir(cls_path)):
                if patient_filter and patient not in patient_filter:
                    continue

                patient_dir = os.path.join(cls_path, patient)
                if not os.path.isdir(patient_dir):
                    continue

                npy_files = sorted([fname for fname in os.listdir(patient_dir)
                                    if fname.endswith(".npy") and "slice_" in fname])

                filtered_files = [
                    fname for fname in npy_files
                    if 30 <= int(fname.split("_")[1].split(".")[0]) <= 65
                ]

                for fname in filtered_files:
                    self.file_list.append((os.path.join(patient_dir, fname), patient, fname))

        if label_dict_path and os.path.exists(label_dict_path):
            with open(label_dict_path, 'r') as f:
                subject_data = json.load(f)
                for patient_id, info in subject_data.items():
                    diagnosis = info.get("Diagnosis")
                    if diagnosis == "AD":
                        self.label_dict[patient_id] = 0
                    elif diagnosis == "CAA_ICH":
                        self.label_dict[patient_id] = 1
                    elif diagnosis == "CAA_CI":
                        self.label_dict[patient_id] = 2

        print(f"✅ 加载完成，共加载 {len(self.file_list)} 个文件。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath, patient, fname = self.file_list[idx]
        data = np.load(filepath)
        data = torch.from_numpy(data).float()

        pad_h = ((128 - data.shape[1]) // 2, (128 - data.shape[1]) - (128 - data.shape[1]) // 2)
        pad_w = ((128 - data.shape[2]) // 2, (128 - data.shape[2]) - (128 - data.shape[2]) // 2)
        pad_d = ((128 - data.shape[3]) // 2, (128 - data.shape[3]) - (128 - data.shape[3]) // 2)
        data = np.pad(data, ((0, 0), pad_h, pad_w, pad_d), mode='constant', constant_values=0)

        label = self.label_dict.get(patient, -1)
        return torch.tensor(data), torch.tensor(label), patient, fname


import pandas as pd  # ✅ 加入 pandas

def predict(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suffix = config["suffix"]

     # 將 Excel 檔案存入 output_dir
    output_ex_dir = "test_all_excel_result"
    os.makedirs(output_ex_dir, exist_ok=True)

    with open("5fold.json", "r") as f:
        fold_data = json.load(f)

    fold_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    # 初始化 Excel 資料
    classification_metrics_data = []
    patient_metrics_data = []

    for model_path, output_path in zip(config["model_paths"], config["predict_output_paths"]):
        newfold_idx = int(model_path.split("_")[-2][-1])
        test_fold_idx = fold_mapping[newfold_idx]
        test_patients = fold_data[str(test_fold_idx)]

        print(f"\n🚀 Running prediction for newfold_{newfold_idx} (5fold index: {test_fold_idx})")
        print(f"Test patients: {test_patients}")

        test_dataset = PredictDataset(
            root_path=config["test_data_path"],
            selected_classes=config["selected_classes"],
            label_dict_path=config["subject_json"],
            patient_filter=test_patients
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = DenseNet201(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            init_features=32,
            growth_rate=16,
            block_config=(2, 2, 2, 2),
            dropout_prob=0.3
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"✅ Loaded state_dict from {model_path}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded model weights directly from {model_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        all_preds, all_labels = [], []
        all_probs = []  # 用於計算 AUC
        patient_summary = defaultdict(lambda: {"true_label": None, "preds": [], "probs": []})

        with open(output_path, "w") as f:
            for data, label, patient_id, fname in test_loader:
                if data.ndim == 4:
                    data = data.unsqueeze(0)
                data = data.to(device)
                label = label.to(device)

                with torch.no_grad():
                    output = model(data)
                    prob = F.softmax(output, dim=1)
                    pred = torch.argmax(prob, dim=1)

                pred_label = pred.item()
                true_label = label.item()
                probabilities = prob.cpu().numpy().flatten().tolist()

                all_preds.append(pred_label)
                all_labels.append(true_label)
                all_probs.append(probabilities[1])  # 假設類別 1 是目標類別

                pid = patient_id[0]
                patient_summary[pid]["true_label"] = true_label
                patient_summary[pid]["preds"].append(pred_label)
                patient_summary[pid]["probs"].append(probabilities)

                result_str = f"{pid}_{fname[0]}: Predicted={pred_label}, True={true_label}, Probabilities={probabilities}"
                f.write(result_str + "\n")

        patient_result_txt_path = os.path.join(output_ex_dir, f"patient_prediction_summary_{suffix}_fold_{newfold_idx}.txt")
        with open(patient_result_txt_path, "w") as f_txt:
            for patient_id, summary in patient_summary.items():
                true_label = summary["true_label"]
                preds = summary["preds"]
                probs = summary["probs"]

                pred_counter = Counter(preds)
                total_slices = len(preds)
                pred_0_count = pred_counter.get(0, 0)
                pred_1_count = pred_counter.get(1, 0)

                # 計算平均 probability
                avg_prob = np.mean(np.array(probs), axis=0).tolist()

                majority_vote = Counter(preds).most_common(1)[0][0]

                f_txt.write(f"Patient: {patient_id}\n")
                f_txt.write(f"True Label: {true_label}\n")
                f_txt.write(f"Predicted (majority): {majority_vote}\n")
                f_txt.write(f"Slice Count: {total_slices}\n")
                f_txt.write(f"Predicted 0 Count: {pred_0_count}\n")
                f_txt.write(f"Predicted 1 Count: {pred_1_count}\n")
                f_txt.write(f"Average Probabilities: {avg_prob}\n\n")
        
                print(f"📝 每位病人的預測細節已存入 {patient_result_txt_path}")


        # ➤ classification_metrics.txt
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()  # 展開混淆矩陣
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 計算 Specificity
        acc=(tp + tn) / (tp+tn+fp+fn) if (tp + tn + fp + fn) > 0 else 0  # 計算 Accuracy
        # acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)  # 計算 AUC

        classification_metrics_data.append({
            
            "Model Path": f"{suffix}_fold_{newfold_idx}",
            "AUC": auc,
            "Confusion Matrix": cm.tolist(),  # 將矩陣轉為列表以便存儲
            
            "Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": prec,
            "F1 Score": f1,
        })

        # ➤ patient_level_metrics.txt
        patient_level_true = []
        patient_level_pred = []

        for patient_id, summary in patient_summary.items():
            preds = summary["preds"]
            majority_vote = Counter(preds).most_common(1)[0][0]
            true_label = summary["true_label"]

            patient_level_true.append(true_label)
            patient_level_pred.append(majority_vote)

        cm_patient = confusion_matrix(patient_level_true, patient_level_pred)
        tn_patient, fp_patient, fn_patient, tp_patient = cm_patient.ravel()  # 展開混淆矩陣
        sensitivity_patient = tp_patient / (tp_patient + fn_patient) if (tp_patient + fn_patient) > 0 else 0  # Sensitivity
        specificity_patient = tn_patient / (tn_patient + fp_patient) if (tn_patient + fp_patient) > 0 else 0  # 計算 Specificity
        accuracy_patient = accuracy_score(patient_level_true, patient_level_pred)
        precision_patient = precision_score(patient_level_true, patient_level_pred, average='weighted', zero_division=0)
        recall_patient = recall_score(patient_level_true, patient_level_pred, average='weighted', zero_division=0)
        f1_patient = f1_score(patient_level_true, patient_level_pred, average='weighted', zero_division=0)
        auc_patient = roc_auc_score(patient_level_true, [summary["probs"][0][1] for summary in patient_summary.values()])  # 使用第一個病人的概率計算 AUC

        patient_metrics_data.append({
            "Model Path": f"{suffix}_fold_{newfold_idx}",
            "AUC": auc_patient,  # 使用相同的 AUC 值
            "Patient-Level Confusion Matrix": cm_patient.tolist(),  # 將矩陣轉為列表以便存儲
          
            "Patient-Level Accuracy": accuracy_patient,
            
            "Patient-Level Sensitivity": sensitivity_patient,
            "Patient-Level Specificity": specificity_patient,
            "Patient-Level Precision": precision_patient,
            "Patient-Level F1 Score": f1_patient
        })
    return classification_metrics_data, patient_metrics_data

    


if __name__ == '__main__':
    all_classification_data = []
    all_patient_data = []

    for config in ALL_CONFIGS:
        cls_data, pat_data = predict(config)
        all_classification_data.extend(cls_data)
        all_patient_data.extend(pat_data)

        # 統整每個 suffix 的 5 folds 結果為 avg（使用 config["suffix"] 判別）
        suffix = config["suffix"]

        cls_subset = [d for d in cls_data if d["Model Path"].startswith(suffix)]
        pat_subset = [d for d in pat_data if d["Model Path"].startswith(suffix)]

        if len(cls_subset) == 5:
            confusion_matrices = [np.array(d["Confusion Matrix"]) for d in cls_subset]
            avg_cm = np.sum(confusion_matrices, axis=0).astype(int).tolist()

            cls_avg = {
                "Model Path": f"{suffix}_all_fold",
                "AUC": np.mean([d["AUC"] for d in cls_subset]),
                "Confusion Matrix": avg_cm,
                "Accuracy": np.mean([d["Accuracy"] for d in cls_subset]),
                "Sensitivity": np.mean([d["Sensitivity"] for d in cls_subset]),
                "Specificity": np.mean([d["Specificity"] for d in cls_subset]),
                "Precision": np.mean([d["Precision"] for d in cls_subset]),
                "F1 Score": np.mean([d["F1 Score"] for d in cls_subset]),
            }
            all_classification_data.append(cls_avg)

        if len(pat_subset) == 5:

            # 提取每個 fold 的病人層級混淆矩陣，並將其轉換為 NumPy 陣列
            patient_confusion_matrices = [np.array(d["Patient-Level Confusion Matrix"]) for d in pat_subset]
            
            # 將所有病人層級混淆矩陣進行總和
            avg_patient_cm = np.sum(patient_confusion_matrices, axis=0).astype(int).tolist()
            pat_avg = {
                "Model Path": f"{suffix}_all_fold",
                "AUC": np.mean([d["AUC"] for d in pat_subset]),
                "Patient-Level Confusion Matrix": avg_patient_cm,
                "Patient-Level Accuracy": np.mean([d["Patient-Level Accuracy"] for d in pat_subset]),
                "Patient-Level Sensitivity": np.mean([d["Patient-Level Sensitivity"] for d in pat_subset]),
                "Patient-Level Specificity": np.mean([d["Patient-Level Specificity"] for d in pat_subset]),
                "Patient-Level Precision": np.mean([d["Patient-Level Precision"] for d in pat_subset]),
                "Patient-Level F1 Score": np.mean([d["Patient-Level F1 Score"] for d in pat_subset]),
            }
            all_patient_data.append(pat_avg)

    # ➜ 寫到同一個 Excel
    output_ex_dir = "test_all_excel_result"
    os.makedirs(output_ex_dir, exist_ok=True)
    excel_file = os.path.join(output_ex_dir, "metrics_summary.xlsx")
    print(f"📊 Saving all metrics to {excel_file}")

    classification_df = pd.DataFrame(all_classification_data)
    patient_df = pd.DataFrame(all_patient_data)

    # 提取 all_fold 的數據
    classification_all_fold = classification_df[classification_df["Model Path"].str.endswith("_all_fold")]
    patient_all_fold = patient_df[patient_df["Model Path"].str.endswith("_all_fold")]

    # 交換 5_12 和 5_24 的數據
    classification_all_fold["Model Path"] = classification_all_fold["Model Path"].str.replace("5_12", "TEMP")
    classification_all_fold["Model Path"] = classification_all_fold["Model Path"].str.replace("5_24", "5_12")
    classification_all_fold["Model Path"] = classification_all_fold["Model Path"].str.replace("TEMP", "5_24")

    patient_all_fold["Model Path"] = patient_all_fold["Model Path"].str.replace("5_12", "TEMP")
    patient_all_fold["Model Path"] = patient_all_fold["Model Path"].str.replace("5_24", "5_12")
    patient_all_fold["Model Path"] = patient_all_fold["Model Path"].str.replace("TEMP", "5_24")

    # 更新標題
    classification_all_fold.rename(columns={"Model Path": "Updated Model Path"}, inplace=True)
    patient_all_fold.rename(columns={"Model Path": "Updated Model Path"}, inplace=True)

    # 保存到 Excel
    output_ex_dir = "test_all_excel_result"
    os.makedirs(output_ex_dir, exist_ok=True)
    excel_file = os.path.join(output_ex_dir, "metrics_summary.xlsx")

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        classification_df.to_excel(writer, sheet_name="Classification Metrics", index=False)
        patient_df.to_excel(writer, sheet_name="Patient Metrics", index=False)
        
        # 保存交換後的數據到新的工作表
        classification_all_fold.to_excel(writer, sheet_name="Classification All Fold", index=False)
        patient_all_fold.to_excel(writer, sheet_name="Patient All Fold", index=False)

    print(f"✅ 所有指標已存入 {excel_file}")
