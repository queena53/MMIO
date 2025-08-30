import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from monai.networks.nets import DenseNet201


def main():
    # 初始化 MONAI 的 DenseNet201 模型
    # model = DenseNet201(
    #     spatial_dims=3,       # 3D 模型
    #     in_channels=1,        # 輸入通道數，例如 MRI 是單通道
    #     out_channels=2,       # 輸出類別數，例如二分類
    #     init_features=32,     # 初始卷積層的特徵數
    #     growth_rate=16,       # 每層的特徵增長率
    #     block_config=(6, 12, 48, 32),  # 每個 Dense Block 的層數
    #     dropout_prob=0.2      # Dropout 機率
    # )

    model = DenseNet201(
        spatial_dims=3,
        in_channels=1,  
        out_channels=2,  # 二分類
        init_features=32,
        growth_rate=16,
        block_config=(2, 2, 2, 2),  # 可以適當調整為 (2,2,2,1) 或 (2,2,1)
        dropout_prob=0.3  # dropout 機率
        )


    model.load_state_dict(torch.load("./output/classification_AD_CAA_mask_frame22_24_slice1/desnet201_FocalLoss/desnet201_FocalLoss_fold_3_model.pth", map_location="cpu"))  # 載入已訓練的模型權重
    model.eval()

    # 設定 Grad-CAM 的目標層
    target_layers = [model.features.denseblock4.denselayer2.layers.conv2]  # 選擇最後一個 Dense Block

    # 定義資料轉換
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 單通道資料的標準化
    ])

    # 載入 3D 資料 (npy 檔案)
    npy_path = "./noich_right_hemisphere_mask_frame22_24_slice1_class_npydata/AD/patient004/slice_40.npy"
    assert os.path.exists(npy_path), f"File '{npy_path}' does not exist."
    volume = np.load(npy_path)  # 形狀: [D, H, W]
    print(f"Loaded volume shape: {volume.shape}")
    volume=np.pad(volume, ((0, 0), (63, 62), (41, 41), (10, 9)), mode='constant', constant_values=0)
    print(f"After padding shape: {volume.shape}")  # shape: [1, 128, 128, 128]

    # 選擇中間切片進行 Grad-CAM
    z = volume.shape[0] // 2  # 中間切片索引
    slice_img = volume[z, :, :]  # 形狀: [H, W]
    norm_slice = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)  # 正規化到 [0, 1]
    slice_img_rgb = (np.stack([norm_slice] * 3, axis=-1) * 255).astype(np.uint8)  # 轉換為 RGB 圖像

    # 準備輸入張量
    input_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).requires_grad_(True)  # [1, 1, D, H, W]

    # 前向傳播
    output = model(input_tensor)  # 模型輸出
    pred = output.argmax(dim=1, keepdim=True)  # 預測類別

    # 選擇對應類別的分數
    selected_output = output.gather(1, pred)

    # 計算 Grad-CAM
    grad_cam = torch.autograd.grad(outputs=selected_output, inputs=input_tensor, 
                                    retain_graph=True, create_graph=True)[0]
    grad_cam = grad_cam.mean(dim=1, keepdim=True)  # 沿通道維度取平均
    grad_cam = torch.relu(grad_cam)  # ReLU 激活
    grad_cam = grad_cam / (grad_cam.max() + 1e-8)  # 正規化避免除以零

    # 提取中間切片的 Grad-CAM
    grayscale_cam = grad_cam.squeeze().detach().cpu().numpy()  # 形狀: [D, H, W]
    grayscale_cam_slice = grayscale_cam[z, :, :]  # 提取中間切片

    # 提取中間切片的 RGB 圖像
    slice_img_rgb_2d = slice_img_rgb[z, :, :]  # 提取中間切片，形狀: [H, W, 3]

    # 將 Grad-CAM 疊加到原始圖像
    visualization = show_cam_on_image(slice_img_rgb_2d.astype(dtype=np.float32) / 255.,
                                    grayscale_cam_slice,
                                    use_rgb=True)

    # 儲存結果
    # 確保目錄存在
    output_dir = "./gradcam_test"
    os.makedirs(output_dir, exist_ok=True)  # 如果目錄不存在則創建

    # 儲存結果
    output_path = os.path.join(output_dir, "gradcam_result.png") 
    plt.title(f"Grad-CAM Visualization (Category: {pred.item()})")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 儲存圖像
    plt.close()  # 關閉圖像以釋放記憶體
    print(f"Grad-CAM 結果已儲存至 {output_path}")

if __name__ == "__main__":
    main()