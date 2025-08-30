import os, json, torch
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from monai.networks.nets import  resnet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import zoom
import cv2


def generate_gradcam_all_layers(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = "gradcam_outputs/frame5_12_resnet/AD/patient004/slice_40"

    os.makedirs(save_root, exist_ok=True)
    selected_frame = config["gradcam_selected_frame"]

    # 載入模型
    model = resnet.resnet18(
        spatial_dims=3,          # 使用 3D 卷積
        n_input_channels=1,    # 輸入通道數為 120
        num_classes=2            # 二分類
        ).to(device)


    model.load_state_dict(torch.load("./output/classification_AD_CAA_frame5_12_slice1/resnet18_CrossEntropy_Losslabel_smoothing/resnet18_CrossEntropy_Losslabel_smoothing_fold_0_model.pth", map_location=device))
    model.eval()

    # 準備輸入
    volume = np.load(config["gradcam_input_npy"])  # [1, 20, 46, 109]
    assert volume.shape == (1, 8, 46, 109)
    volume = np.pad(volume, ((0,0), (60,60), (41,41), (10,9)), mode='constant', constant_values=0)
    input_tensor = torch.tensor(volume).float().to(device).unsqueeze(0)  # [1, 1, 128, 128, 128]
    padded_volume = input_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # 搜尋可視化層
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            target_layers.append((name, module))

    print(f"✅ Found {len(target_layers)} Conv3d layers")

    for name, layer in target_layers:
        print(f"\n🔍 Running Grad-CAM for layer: {name}")
        layer_dir = os.path.join(save_root, name.replace(".", "_"))
        os.makedirs(layer_dir, exist_ok=True)

        cam = GradCAM(model=model, target_layers=[layer])

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        grayscale_cam = grayscale_cam[60:-60, 41:-41, 10:-9]  # 回復 padding 前形狀
        volume_unpad = padded_volume[60:-60, 41:-41, 10:-9]    # shape: [24, 46, 109]

        gif_frames = []
        for d in range(8):  # 可視化前 20 slice
            slice_img = volume_unpad[d, :, :]
            slice_img_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)
            slice_img_rgb = (np.stack([slice_img_norm] * 3, axis=-1) * 255).astype(np.uint8)
            cam_slice = grayscale_cam[d, :, :]

            scale_factors = (
                slice_img_rgb.shape[0] / cam_slice.shape[0],
                slice_img_rgb.shape[1] / cam_slice.shape[1]
            )
            cam_resized = cv2.resize(cam_slice, (slice_img_rgb.shape[1], slice_img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


            overlay = show_cam_on_image(slice_img_rgb.astype(np.float32)/255.0, cam_resized, use_rgb=True)
            pil_img = Image.fromarray(overlay)
            gif_frames.append(pil_img)

            if d == selected_frame:
                pil_img.save(os.path.join(layer_dir, f"frame_{d}.png"))
                raw_img = Image.fromarray(slice_img_rgb)
                raw_img.save(os.path.join(layer_dir, f"frame_{d}_raw.png"))


        

        # 儲存動畫
        gif_frames_with_text = []
        for d, frame in enumerate(gif_frames):
            # 在每張圖像上添加文字
            draw = ImageDraw.Draw(frame)
            text = f"Frame: {d+5}"
            font = ImageFont.load_default()  # 使用默認字體

            # 計算文字邊界框
            text_bbox = draw.textbbox((0, 0), text, font=font)  # 返回 (left, top, right, bottom)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 計算左下角的文字位置
            text_x = 3  # 距離左邊緣 10 像素
            text_y = frame.height - text_height - 3  # 距離下邊緣 10 像素

            # 在圖像左上角添加文字
            draw.text((text_x, text_y), text, fill="white", font=font)
            gif_frames_with_text.append(frame)

        # 儲存動畫
        gif_path = os.path.join(layer_dir, f"{name.replace('.', '_')}_animation.gif")
        gif_frames_with_text[0].save(
            gif_path, save_all=True, append_images=gif_frames_with_text[1:], duration=200, loop=0
        )
        print(f"💾 Saved Grad-CAM animation to {gif_path}")

if __name__ == '__main__':
    generate_gradcam_all_layers("config.json")
