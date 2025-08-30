import os
import torch
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from monai.networks.nets import DenseNet201
from utils import GradCAM, show_cam_on_image

def generate_gradcam_single_npy(model, input_path, output_path, selected_frame, device):
    volume = np.load(input_path)
    print(f"Loaded volume shape: {volume.shape}")

    target_shape = (128, 128, 128)
    padded = np.zeros(target_shape, dtype=np.float32)
    z, y, x = volume.shape[1:]
    zs, ys, xs = (target_shape[0]-z)//2, (target_shape[1]-y)//2, (target_shape[2]-x)//2
    padded[zs:zs+z, ys:ys+y, xs:xs+x] = volume[0]
    input_tensor = torch.tensor(padded).unsqueeze(0).unsqueeze(0).to(device)

    target_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Conv3d)]

    for name, layer in target_layers:
        print(f"🔍 Running Grad-CAM for layer: {name}")
        # layer_dir = os.path.join(output_path, name.replace(".", "_"))
        # os.makedirs(layer_dir, exist_ok=True)

        with GradCAM(model=model, target_layers=[layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]

        cam_unpad = grayscale_cam[zs:zs+z, ys:ys+y, xs:xs+x]
        volume_unpad = volume[0]

        filename = os.path.basename(input_path)
        print(f"Processing file: {filename}")
        patient_id = os.path.basename(os.path.dirname(input_path))
        print(f"Patient ID: {patient_id}")

        parts = filename.split('_')
        if len(parts) == 2 and parts[0] == "slice":
            slice_index = parts[1].replace('.npy', '')
        else:
            raise ValueError(f"Unexpected file name format: {filename}")

        for d in range(volume_unpad.shape[0]):
            slice_img = volume_unpad[d]
            norm_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
            slice_rgb = (np.stack([norm_slice] * 3, axis=-1) * 255).astype(np.uint8)

            cam_slice = cam_unpad[d]
            cam_resized = cv2.resize(cam_slice, (slice_rgb.shape[1], slice_rgb.shape[0]))
            slice_rgb_norm = slice_rgb.astype(np.float32) / 255.0
            overlay = show_cam_on_image(slice_rgb_norm, cam_resized, use_rgb=True)

            layer_name = name.replace(".", "_")
            output_filename = f"{patient_id}_slice_{slice_index}_frame{d}_{layer_name}.png"

            # 🔧 新增：儲存到 patient 子資料夾
            output_patient_dir = os.path.join(output_path, patient_id)
            os.makedirs(output_patient_dir, exist_ok=True)
            output_filepath = os.path.join(output_patient_dir, output_filename)


            # output_filepath = os.path.join(output_path, output_filename)
            Image.fromarray(overlay).save(output_filepath, quality=95)
            print(f"💾 Saved: {output_filepath}")

def run_batch_gradcam():
    test_data_paths = [
   
    
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_occipital_mask_frame5_12_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_occipital_mask_frame5_24_slice1_class_npydata",
    "./SUVr_pons_classified/noich_right_hemisphere_suvr_pons_occipital_mask_frame22_24_slice1_class_npydata",
    ]

    selected_frame = 10
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ✅ 載入 5fold.json
    with open("5fold.json", "r") as f:
        fold_data = json.load(f)

    for test_data_path in test_data_paths:
        base_name = os.path.basename(test_data_path)
        suffix = base_name.replace("noich_right_hemisphere_", "").replace("_class_npydata", "")
        model_dir = f"./output/classification_newfold_AD_CAA_{suffix}/desnet201_ne"
        print(f"Processing test data path: {test_data_path}")
        print(f"Selected frame: {selected_frame}")

        model_paths = [
            os.path.join(model_dir, f"desnet201_newfold_{i}_model.pth") for i in range(5)
        ]

        for fold_idx, model_path in enumerate(model_paths):
            print(f"📂 Processing fold index: {fold_idx}")
            print(f"📂 Model path: {model_path}")
            output_root = os.path.join("./gradcam_all__output", f"{suffix}_fold{fold_idx}")
            print(f"📂 Output directory: {output_root}")

            model = DenseNet201(
                spatial_dims=3, in_channels=1, out_channels=2,
                init_features=32, growth_rate=16, block_config=(2, 2, 2, 2), dropout_prob=0.3
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # ✅ 取得當前 fold 對應的病人清單
            current_fold_patients = set(fold_data[str(fold_idx)])
            # print(f"📋 Current fold patients: {current_fold_patients}")

            for category in os.listdir(test_data_path):
                if category not in ["AD", "CAA_ICH"]:
                    continue

                category_dir = os.path.join(test_data_path, category)
                if not os.path.isdir(category_dir):
                    continue

                for patient in os.listdir(category_dir):
                    if patient not in current_fold_patients:  # ✅ 檢查是否屬於目前 fold
                        continue

                    patient_dir = os.path.join(category_dir, patient)
                    if not os.path.isdir(patient_dir):
                        continue

                    for file in os.listdir(patient_dir):
                        if file.endswith(".npy") and file.startswith("slice_"):
                            slice_index = int(file.split("_")[1].replace(".npy", ""))
                            if slice_index < 30 or slice_index > 60:
                                continue

                            npy_path = os.path.join(patient_dir, file)
                            os.makedirs(output_root, exist_ok=True)
                            generate_gradcam_single_npy(model, npy_path, output_root, selected_frame, device)

if __name__ == '__main__':
    run_batch_gradcam()
