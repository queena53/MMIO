import os
import re
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


# ========= 你要處理的所有資料夾清單 =========
file_list = [
    "./val_all_result/val_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_all_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_all_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_frontal_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_occipital_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_parietal_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame22_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame22_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame22_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame22_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame22_24_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_12_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_12_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_12_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_12_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_12_slice1_fold_4.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_24_slice1_fold_0.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_24_slice1_fold_1.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_24_slice1_fold_2.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_24_slice1_fold_3.txt",
    "./val_all_result/val_suvr_pons_temporal_mask_frame5_24_slice1_fold_4.txt"

    ]

output_dir = "val_patient_analysis_slice_5fold_summary"
os.makedirs(output_dir, exist_ok=True)
output_excel = os.path.join(output_dir, "all_models_5fold_analysis.xlsx")

# 分組
model_groups = defaultdict(list)
for f in file_list:
    model_title = re.sub(r'_fold_\d+\.txt$', '', os.path.basename(f))
    model_groups[model_title].append(f)

# 正則式
pattern = re.compile(
    r'(?P<filename>(?P<patient>patient\d+)_?(?P<slice>slice_\d+)\.npy): Predicted=(?P<pred>\d), True=(?P<true>\d), '
    r'Probabilities=\[(?P<prob0>[\d\.eE+-]+), (?P<prob1>[\d\.eE+-]+)\]'
)

all_top_summary = []

with pd.ExcelWriter(output_excel) as writer:
    for model_title, paths in model_groups.items():
        slice_map = defaultdict(list)

        for path in paths:
            with open(path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        slice_name = match.group("slice")
                        prob0 = float(match.group("prob0"))
                        prob1 = float(match.group("prob1"))
                        pred = int(match.group("pred"))
                        true = int(match.group("true"))
                        correct = int(pred == true)
                        confidence = max(prob0, prob1)

                        slice_map[slice_name].append({
                            "true": true,
                            "pred": pred,
                            "prob0": prob0,
                            "prob1": prob1,
                            "correct": correct,
                            "confidence": confidence
                        })

        summary = []
        for slice_name, records in slice_map.items():
            trues = [r["true"] for r in records]
            preds = [r["pred"] for r in records]
            prob1s = [r["prob1"] for r in records]
            prob0s = [r["prob0"] for r in records]
            corrects = [r["correct"] for r in records]
            confs = [r["confidence"] for r in records]

            try:
                tn, fp, fn, tp = confusion_matrix(trues, preds, labels=[0, 1]).ravel()
            except:
                tn = fp = fn = tp = 0

            try:
                auc = round(roc_auc_score(trues, prob1s), 4)
            except:
                auc = None
            try:
                acc = round(accuracy_score(trues, preds), 4)
            except:
                acc = None

            sensitivity = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
            specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else None

            summary.append({
                'slice': slice_name,
                'mean_prob0': round(sum(prob0s) / len(prob0s), 4),
                'mean_prob1': round(sum(prob1s) / len(prob1s), 4),
                'accuracy': round(sum(corrects) / len(corrects), 4),
                'mean_confidence': round(sum(confs) / len(confs), 4),
                'count': len(records),
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'AUC': auc,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Overall_Accuracy': acc
            })

        df_summary = pd.DataFrame(summary)
        df_summary = df_summary.sort_values(by='Overall_Accuracy', ascending=False, na_position='last')

        # ✅ 儲存當前模型的詳細 sheet
        sheet_name = f"{model_title}_summary"[:31]
        df_summary.to_excel(writer, index=False, sheet_name=sheet_name)

        # ✅ 提取 top3/5/10
        for topN, count in zip(["top3", "top5", "top10"], [3, 5, 10]):
            for i, row in df_summary.head(count).iterrows():
                all_top_summary.append({
                    "model": model_title,
                    "topN": topN,
                    "slice": row["slice"],
                    "Overall_Accuracy": row["Overall_Accuracy"],
                    "AUC": row["AUC"],
                    "Sensitivity": row["Sensitivity"],
                    "Specificity": row["Specificity"],
                    "mean_confidence": row["mean_confidence"],
                    "count": row["count"]
                })

    # ✅ 儲存 top3/5/10 統整表
    df_top = pd.DataFrame(all_top_summary)
        # 按照 topN 的順序進行排序，確保順序為 top3、top5、top10
    df_top['topN_order'] = df_top['topN'].map({"top3": 1, "top5": 2, "top10": 3})  # 新增排序欄位
    df_top = df_top.sort_values(by=["model", "topN_order", "Overall_Accuracy"], ascending=[True, True, False])  # 排序

    # 刪除輔助排序欄位
    df_top = df_top.drop(columns=["topN_order"])

    # 將結果保存到 Excel
    df_top.to_excel(writer, index=False, sheet_name="best_top_slices")

print(f"✅ 完成！每個模型的 slice 分析與 top3/5/10 已儲存到：{output_excel}")