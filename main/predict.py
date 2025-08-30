import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from model_medicalNet import resnet10, resnet18, resnet101
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter


class PredictDataset(Dataset):
    def __init__(self, root_path, selected_classes, label_dict_path=None):
        self.file_list = []
        self.label_dict = {}

        # 遍历 selected_classes 中的类别
        for cls in selected_classes:
            cls_path = os.path.join(root_path, cls)
            if not os.path.isdir(cls_path):
                print(f"⚠️ 类别文件夹不存在: {cls_path}")
                continue

            # 遍历类别文件夹中的病人文件夹
            for patient in sorted(os.listdir(cls_path)):
                patient_dir = os.path.join(cls_path, patient)
                if not os.path.isdir(patient_dir):
                    continue
                npy_files = sorted([fname for fname in os.listdir(patient_dir) 
                                    if fname.endswith(".npy") and "slice_" in fname])

                # 只处理 slice_30 到 slice_65 的文件
                filtered_files = [
                    fname for fname in npy_files 
                    if 30 <= int(fname.split("_")[1].split(".")[0]) <= 65
                ]
                print(f"🔍 正在加载病人: {patient}, 文件数: {len(filtered_files)}")

                for fname in filtered_files:
                    self.file_list.append((os.path.join(patient_dir, fname), patient, fname))

        # 加载标签字典
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
        data = np.load(filepath)  # shape: [1, 24, 46, 109]
        data = torch.from_numpy(data).float()

        # 填充到 [1, 128, 128, 128]
        data = np.pad(data, ((0, 0), (54, 54), (41, 41), (10, 9)), mode='constant', constant_values=0)
        patient_id = patient  # patient 文件夹名就是 patient_id
        label = self.label_dict.get(patient_id, -1)
        label = torch.tensor(label, dtype=torch.long)

        return data, label, patient_id, fname


def predict():
    import collections
    from collections import defaultdict, Counter

    # 读取 config
    with open('config.json') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试数据集
    test_dataset = PredictDataset(
        root_path=config["test_data_path"],
        selected_classes=config["selected_classes"],
        label_dict_path=config.get("subject_json")
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = DenseNet201(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_features=32,
        growth_rate=16,
        block_config=(2, 2, 2, 2),
        dropout_prob=0.3
    ).to(device)

    # 加载 checkpoint
    checkpoint_path = config["model_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"✅ Loaded state_dict from {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded model weights directly from {checkpoint_path}")

    # 输出预测结果
    os.makedirs(config["predict_output_path"], exist_ok=True)
    output_file = os.path.join(config["predict_output_path"], "prediction_results.txt")

    all_preds = []
    all_labels = []
    all_probs = []

    patient_summary = defaultdict(lambda: {
        "true_label": None,
        "preds": [],
        "probs": []
    })

    with open(output_file, "w") as f:
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
            all_probs.append(probabilities)

            patient_key = patient_id[0]
            patient_summary[patient_key]["true_label"] = true_label
            patient_summary[patient_key]["preds"].append(pred_label)
            patient_summary[patient_key]["probs"].append(probabilities)

            result_str = (f"{patient_key}_{fname[0]}: Predicted={pred_label}, True={true_label}, "
                          f"Probabilities={probabilities}")
            print(result_str)
            f.write(result_str + "\n")

    # 整体评估指标（基于切片）
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics_file = os.path.join(config["predict_output_path"], "classification_metrics.txt")
    with open(metrics_file, "w") as mf:
        mf.write("📊 Slice-Level Classification Metrics:\n")
        mf.write(f"Confusion Matrix:\n{cm}\n")
        mf.write(f"Accuracy: {accuracy:.4f}\n")
        mf.write(f"Precision: {precision:.4f}\n")
        mf.write(f"Recall: {recall:.4f}\n")
        mf.write(f"F1 Score: {f1:.4f}\n")
    print(f"📁 切片级分类指标保存至：{metrics_file}")

    # 每个病人的多数投票统计
    summary_file = os.path.join(config["predict_output_path"], "patient_summary.txt")
    with open(summary_file, "w") as sf:
        for patient_id, summary in patient_summary.items():
            preds = summary["preds"]
            probs = np.array(summary["probs"])
            majority_vote = Counter(preds).most_common(1)[0][0]
            avg_probs = np.mean(probs, axis=0).tolist()

            sf.write(f"Patient: {patient_id}\n")
            sf.write(f"True Label: {summary['true_label']}\n")
            sf.write(f"Predicted (majority): {majority_vote}\n")
            sf.write(f"Slice Count: {len(preds)}\n")
            sf.write(f"Predicted 0 Count: {preds.count(0)}\n")
            sf.write(f"Predicted 1 Count: {preds.count(1)}\n")
            sf.write(f"Average Probabilities: {avg_probs}\n")
            sf.write("\n")
    print(f"📁 每个病人汇总结果已保存至: {summary_file}")

    # 病人级别混淆矩阵
    patient_true_labels = []
    patient_majority_preds = []

    for patient_id, summary in patient_summary.items():
        majority_vote = Counter(summary["preds"]).most_common(1)[0][0]
        true_label = summary["true_label"]
        patient_majority_preds.append(majority_vote)
        patient_true_labels.append(true_label)

    cm_patient = confusion_matrix(patient_true_labels, patient_majority_preds)
    accuracy_patient = accuracy_score(patient_true_labels, patient_majority_preds)
    precision_patient = precision_score(patient_true_labels, patient_majority_preds, average='weighted', zero_division=0)
    recall_patient = recall_score(patient_true_labels, patient_majority_preds, average='weighted', zero_division=0)
    f1_patient = f1_score(patient_true_labels, patient_majority_preds, average='weighted', zero_division=0)

    patient_metrics_file = os.path.join(config["predict_output_path"], "patient_level_metrics.txt")
    with open(patient_metrics_file, "w") as pf:
        pf.write("📊 Patient-Level Classification Metrics (majority voting):\n")
        pf.write(f"Confusion Matrix:\n{cm_patient}\n")
        pf.write(f"Accuracy: {accuracy_patient:.4f}\n")
        pf.write(f"Precision: {precision_patient:.4f}\n")
        pf.write(f"Recall: {recall_patient:.4f}\n")
        pf.write(f"F1 Score: {f1_patient:.4f}\n")

    print(f"📁 病人级别混淆矩阵已保存至：{patient_metrics_file}")

    import collections
    from collections import defaultdict, Counter

    # 读取 config
    with open('config.json') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试数据集
    test_dataset = PredictDataset(
        root_path=config["test_data_path"],
        selected_classes=config["selected_classes"],
        label_dict_path=config.get("subject_json")
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = DenseNet201(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # 二分类
        init_features=32,
        growth_rate=16,
        block_config=(2, 2, 2, 2),
        dropout_prob=0.3
    ).to(device)

    # 加载 checkpoint
    checkpoint_path = config["model_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"✅ Loaded state_dict from {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded model weights directly from {checkpoint_path}")

    # 输出预测结果
    os.makedirs(config["predict_output_path"], exist_ok=True)
    output_file = os.path.join(config["predict_output_path"], "prediction_results.txt")

    all_preds = []
    all_labels = []
    all_probs = []

    # ⬇️ 新增：用于病人级统计
    patient_summary = defaultdict(lambda: {
        "true_label": None,
        "preds": [],
        "probs": []
    })

    with open(output_file, "w") as f:
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
            all_probs.append(probabilities)

            patient_key = patient_id[0]
            patient_summary[patient_key]["true_label"] = true_label
            patient_summary[patient_key]["preds"].append(pred_label)
            patient_summary[patient_key]["probs"].append(probabilities)

            result_str = (f"{patient_key}_{fname[0]}: Predicted={pred_label}, True={true_label}, "
                          f"Probabilities={probabilities}")
            print(result_str)
            f.write(result_str + "\n")

    # 计算分类指标（整体）
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # 保存整体分类指标
    metrics_file = os.path.join(config["predict_output_path"], "classification_metrics.txt")
    with open(metrics_file, "w") as mf:
        mf.write("📊 Classification Metrics:\n")
        mf.write(f"Confusion Matrix:\n{cm}\n")
        mf.write(f"Accuracy: {accuracy:.4f}\n")
        mf.write(f"Precision: {precision:.4f}\n")
        mf.write(f"Recall: {recall:.4f}\n")
        mf.write(f"F1 Score: {f1:.4f}\n")
    print(f"📁 分类指标保存至：{metrics_file}")

    # ⬇️ 新增：每个病人的结果汇总
    summary_file = os.path.join(config["predict_output_path"], "patient_summary.txt")
    with open(summary_file, "w") as sf:
        for patient_id, summary in patient_summary.items():
            preds = summary["preds"]
            probs = np.array(summary["probs"])
            majority_vote = Counter(preds).most_common(1)[0][0]
            avg_probs = np.mean(probs, axis=0).tolist()

            sf.write(f"Patient: {patient_id}\n")
            sf.write(f"True Label: {summary['true_label']}\n")
            sf.write(f"Predicted (majority): {majority_vote}\n")
            sf.write(f"Slice Count: {len(preds)}\n")
            sf.write(f"Predicted 0 Count: {preds.count(0)}\n")
            sf.write(f"Predicted 1 Count: {preds.count(1)}\n")
            sf.write(f"Average Probabilities: {avg_probs}\n")
            sf.write("\n")

    print(f"📁 每个病人汇总结果已保存至: {summary_file}")

if __name__ == '__main__':
    predict()