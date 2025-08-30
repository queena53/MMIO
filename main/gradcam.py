import os, json
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from monai.networks.nets import DenseNet201
from utils import GradCAMPlusPlus, show_cam_on_image

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

        with GradCAMPlusPlus(model=model, target_layers=[layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]

        cam_unpad = grayscale_cam[zs:zs+z, ys:ys+y, xs:xs+x]
        volume_unpad = volume[0]

        gif_frames = []
        for d in range(3):
            slice_img = volume_unpad[d]
            norm_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
            slice_rgb = (np.stack([norm_slice]*3, axis=-1) * 255).astype(np.uint8)

            cam_slice = cam_unpad[d]
            cam_resized = cv2.resize(cam_slice, (slice_rgb.shape[1], slice_rgb.shape[0]))
            slice_rgb_norm = slice_rgb.astype(np.float32) / 255.0
            overlay = show_cam_on_image(slice_rgb_norm, cam_resized, use_rgb=True)

            pil_img = Image.fromarray(overlay)
            gif_frames.append(pil_img)

            if d == selected_frame:
                pil_img.save(os.path.join(layer_dir, f"frame_{d}.png"))
                Image.fromarray(slice_rgb).save(os.path.join(layer_dir, f"frame_{d}_raw.png"))

        # 加文字與儲存動畫
        gif_with_text = []
        for i, frame in enumerate(gif_frames):
            draw = ImageDraw.Draw(frame)
            font = ImageFont.load_default()
            text = f"Frame: {i+5}"
            draw.text((3, frame.height - 15), text, fill="white", font=font)
            gif_with_text.append(frame)

        gif_path = os.path.join(layer_dir, f"{name.replace('.', '_')}_animation.gif")
        gif_with_text[0].save(gif_path, save_all=True, append_images=gif_with_text[1:], duration=200, loop=0)
        print(f"💾 Saved Grad-CAM animation to {gif_path}")

def generate_gradcam_all_npy(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    input_root = config["gradcam_input_root"]
    output_root = config["gradcam_output_root"]
    selected_frame = config["gradcam_selected_frame"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = DenseNet201(
        spatial_dims=3, in_channels=1, out_channels=2,
        init_features=32, growth_rate=16, block_config=(2, 2, 2, 2), dropout_prob=0.3
    ).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    # 遍歷所有 .npy 檔案
    for patient in os.listdir(input_root):
        patient_dir = os.path.join(input_root, patient)
        if not os.path.isdir(patient_dir):
            continue
        for file in os.listdir(patient_dir):
            if file.endswith(".npy"):
                npy_path = os.path.join(patient_dir, file)
                rel_path = os.path.relpath(npy_path, input_root)
                output_path = os.path.join(output_root, os.path.dirname(rel_path), os.path.splitext(file)[0])
                print(f"\n🚀 Processing: {npy_path}")
                generate_gradcam_single_npy(model, npy_path, output_path, selected_frame, device)

if __name__ == '__main__':
    generate_gradcam_all_npy("config.json")
