import os, json
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from monai.networks.nets import DenseNet201
from utils import GradCAM, show_cam_on_image

def generate_gradcam_single_npy(model, input_path, output_path, selected_frame, device):
    volume = np.load(input_path)  # shape: (1, 3, 46, 109)
    assert volume.shape == (1, 20, 46, 109)

    # padding to 128³
    target_shape = (128, 128, 128)
    padded = np.zeros(target_shape, dtype=np.float32)
    z, y, x = volume.shape[1:]
    zs, ys, xs = (target_shape[0]-z)//2, (target_shape[1]-y)//2, (target_shape[2]-x)//2
    padded[zs:zs+z, ys:ys+y, xs:xs+x] = volume[0]
    input_tensor = torch.tensor(padded).unsqueeze(0).unsqueeze(0).to(device)

    target_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Conv3d)]

    for name, layer in target_layers:
        print(f"🔍 Running Grad-CAM for layer: {name}")
        layer_dir = os.path.join(output_path, name.replace(".", "_"))
        os.makedirs(layer_dir, exist_ok=True)

        with GradCAM(model=model, target_layers=[layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]

        cam_unpad = grayscale_cam[zs:zs+z, ys:ys+y, xs:xs+x]
        volume_unpad = volume[0]

        gif_frames = []

        # 修正檔案命名邏輯
        filename = os.path.basename(input_path)  # 獲取檔案名
        print(f"Processing file: {filename}")
        # 提取 patient_id
        patient_id = os.path.basename(os.path.dirname(input_path))  # 取得父目錄名稱 (如 patient043)

        parts = filename.split('_')  # 分割檔案名
        if len(parts) == 2 and parts[0] == "slice":  # 適應格式為 slice_52.npy
            slice_index = parts[1].replace('.npy', '')  # 提取切片索引，例如 "52"
        else:
            raise ValueError(f"Unexpected file name format: {filename}")
        for d in range(3):
            slice_img = volume_unpad[d]
            norm_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
            slice_rgb = (np.stack([norm_slice] * 3, axis=-1) * 255).astype(np.uint8)

            cam_slice = cam_unpad[d]
            cam_resized = cv2.resize(cam_slice, (slice_rgb.shape[1], slice_rgb.shape[0]))  # 調整大小為 slice_rgb 的大小
            slice_rgb_norm = slice_rgb.astype(np.float32) / 255.0
            overlay = show_cam_on_image(slice_rgb_norm, cam_resized, use_rgb=True)

            layer_name = name.replace(".", "_")  # 替換層名稱中的點
            output_filename = f"{patient_id}_slice_{slice_index}_frame{d}_{layer_name}.png"
            output_filepath = os.path.join(output_path, output_filename)
            Image.fromarray(overlay).save(output_filepath, quality=95)  # 儲存高品質影像
            print(f"💾 Saved high-resolution Grad-CAM overlay to {output_filepath}")

        # 加文字與儲存動畫
        gif_with_text = []
        for i, frame in enumerate(gif_frames):
            draw = ImageDraw.Draw(frame)
            font = ImageFont.load_default()
            text = f"Frame: {i+5}"
            draw.text((3, frame.height - 15), text, fill="white", font=font)
            gif_with_text.append(frame)

        gif_path = os.path.join(layer_dir, f"{name.replace('.', '_')}_animation.gif")
        # gif_with_text[0].save(gif_path, save_all=True, append_images=gif_with_text[1:], duration=200, loop=0)
        # print(f"💾 Saved Grad-CAM animation to {gif_path}")

def generate_gradcam_all_npy(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    input_root = config["gradcam_input_root"]
    output_root = config["gradcam_output_root"]
    selected_frame = 10
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = DenseNet201(
        spatial_dims=3, in_channels=1, out_channels=2,
        init_features=32, growth_rate=16, block_config=(2, 2, 2, 2), dropout_prob=0.3
    ).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    # 遍歷所有類別資料夾 (只處理 AD 和 CAA_ICH)
    for category in os.listdir(input_root):
        if category not in ["AD", "CAA_ICH"]:  # 只處理 AD 和 CAA_ICH
            continue

        category_dir = os.path.join(input_root, category)
        if not os.path.isdir(category_dir):
            continue

        for patient in os.listdir(category_dir):
            patient_dir = os.path.join(category_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            for file in os.listdir(patient_dir):
                if file.endswith(".npy"):
                    parts = file.split('_')
                    if len(parts) == 2 and parts[0] == "slice":
                        slice_index = int(parts[1].replace('.npy', ''))
                        # 只處理 slice30 到 slice60
                        if slice_index < 30 or slice_index > 60:
                            continue
                    else:
                        continue
                    npy_path = os.path.join(patient_dir, file)
                    # 修改 output_path 的生成邏輯
                    output_path = output_root  # 直接使用 output_root 作為基礎路徑
                    os.makedirs(output_path, exist_ok=True)
                    print(f"\n🚀 Processing: {npy_path}")
                    generate_gradcam_single_npy(model, npy_path, output_path, selected_frame, device)

if __name__ == '__main__':
    generate_gradcam_all_npy("config.json")
