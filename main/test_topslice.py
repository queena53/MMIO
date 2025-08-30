import re
import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

val_best_excel = "./val_patient_analysis_slice_5fold_summary/val_all_models_5fold_analysis.xlsx"
if not os.path.exists(val_best_excel):
    raise FileNotFoundError(f"文件不存在：{val_best_excel}")

best_slice_df = pd.read_excel(val_best_excel, sheet_name="best_top_slices")
best_slice_df["model"] = best_slice_df["model"].str.replace(r"^val_", "test_", regex=True)



# ========= 你要處理的所有資料夾清單 =========
file_list = [
    "./test_all_result/test_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_all_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_frontal_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_occipital_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_parietal_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_posteriorcingulate_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_cerebellum_temporal_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_all_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_frontal_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_occipital_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_parietal_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_posteriorcingulate_mask_frame5_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame22_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame22_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame22_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame22_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame22_24_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_12_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_12_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_12_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_12_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_12_slice1_fold_4.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_24_slice1_fold_0.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_24_slice1_fold_1.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_24_slice1_fold_2.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_24_slice1_fold_3.txt",
    "./test_all_result/test_suvr_pons_temporal_mask_frame5_24_slice1_fold_4.txt"

    ]

# 輸出目錄
output_dir = "test_patient_analysis_slice_5fold_summary"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在
output_excel = os.path.join(output_dir, "test_all_models_5fold_analysis.xlsx")

# 分組（將相同模型不同 fold 整合）
model_groups = defaultdict(list)
for f in file_list:
    model_title = re.sub(r'_fold_\d+\.txt$', '', os.path.basename(f))
    model_groups[model_title].append(f)

# 解析 log 檔格式
pattern = re.compile(
    r'(?P<filename>(?P<patient>patient\d+)_?(?P<slice>slice_\d+)\.npy): '
    r'Predicted=(?P<pred>\d), True=(?P<true>\d), '
    r'Probabilities=\[(?P<prob0>[\d\.eE+-]+), (?P<prob1>[\d\.eE+-]+)\]'
)

# 儲存所有模型的分析結果
all_top_summary = []

# 開始分析
with pd.ExcelWriter(output_excel, mode='w', engine='openpyxl') as writer:

    for model_title, paths in model_groups.items():
        slice_map = defaultdict(list)

        # 讀取每個 fold 的預測結果
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

        # 彙整每個切片的統計資訊
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

        
        
        df_summary.to_excel(writer, index=False, sheet_name=f"{model_title[10:]}_summary"[:31])
        print(f"✅ {model_title} 的統計數據已保存到工作表 {model_title[10:]}_summary")

        # ✅ 根據 best_top_slices 指定的切片進行彙整
        model_best_slices = best_slice_df[best_slice_df["model"] == model_title]
        for _, row in model_best_slices.iterrows():
            slice_name = row["slice"]
            matched_summary = df_summary[df_summary["slice"] == slice_name]
            if matched_summary.empty:
                continue  # 找不到該 slice 的資料就跳過

            best_row = matched_summary.iloc[0]
            all_top_summary.append({
                "model": model_title,
                "topN": row["topN"],
                "slice": slice_name,
                "Overall_Accuracy": best_row["Overall_Accuracy"],
                "AUC": best_row["AUC"],
                "TP": best_row["TP"],
                "FP": best_row["FP"],
                "TN": best_row["TN"],
                "FN": best_row["FN"],
                "Sensitivity": best_row["Sensitivity"],
                "Specificity": best_row["Specificity"],
                "mean_confidence": best_row["mean_confidence"],
                "count": best_row["count"]
            })

    # ✅ 最後更新 best_top_slices 表
    df_top = pd.DataFrame(all_top_summary)
    df_top.to_excel(writer, index=False, sheet_name="best_top_slices")

    # ✅ 計算每個模型的 top3、top5、top10 統計數據
    model_summary_stats = []

    for model_title in df_top["model"].unique():
        model_data = df_top[df_top["model"] == model_title]
        for topN in ["top3", "top5", "top10"]:
            topN_data = model_data[model_data["topN"] == topN]
            if not topN_data.empty:
                model_summary_stats.append({
                    "model": model_title,
                    "topN": topN,
                    "mean_Accuracy": round(topN_data["Overall_Accuracy"].mean(), 4),
                    "sum_TP": topN_data["TP"].sum(),
                    "sum_FP": topN_data["FP"].sum(),
                    "sum_TN": topN_data["TN"].sum(),
                    "sum_FN": topN_data["FN"].sum(),
                    "mean_AUC": round(topN_data["AUC"].mean(), 4),
                    "mean_Sensitivity": round(topN_data["Sensitivity"].mean(), 4),
                    "mean_Specificity": round(topN_data["Specificity"].mean(), 4),
                    "mean_confidence": round(topN_data["mean_confidence"].mean(), 4)
                })

    # ✅ 將所有模型的統計數據保存到同一個工作表
    df_model_summary = pd.DataFrame(model_summary_stats)
    df_model_summary.to_excel(writer, index=False, sheet_name="model_summary")

    # ✅ 新增總結工作表
    summary_stats = df_model_summary.groupby("topN").agg({
        "mean_Accuracy": "mean",
        "sum_TP": "sum",
        "sum_FP": "sum",
        "sum_TN": "sum",
        "sum_FN": "sum",
        "mean_AUC": "mean",
        "mean_Sensitivity": "mean",
        "mean_Specificity": "mean",
        "mean_confidence": "mean"
    }).reset_index()

    summary_stats.to_excel(writer, index=False, sheet_name="summary")

print(f"✅ 完成！所有模型的統計數據已保存到同一個工作表，並新增總結工作表。")