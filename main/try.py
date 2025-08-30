import numpy as np
import os
import matplotlib.pyplot as plt
import imageio  # 添加 imageio 模块的导入

import nibabel as nib

# # 加載 .npy 文件
# file_path = 'dataset/AD/patient031/slice_40.npy'
# file_path2 = 'npydata/AD/patient031/slice_40.npy'

# data = np.load(file_path)
# data2 = np.load(file_path2) 

# # data2 = data2.reshape(data2.shape[0] * data2.shape[1], data2.shape[3], data2.shape[2])

# # 打印數據的形狀
# print(data.shape)
# print(data2.shape)
# # print(data)
# print(data2)
# print(np.max(data))

# print(np.max(data2))
# 定义目录
# directory = 'npydata/CAA_ICH/patient089'
# file='normalized_data/patient089/normalized_img/normalized_patient089_pr_pet_norm_frame03.nii'
# file=r'./noich_right_hemisphere_frame5_slice1_class_npydata/AD/patient004/slice_19.npy'
# output_gif_path = r'./tryoutput'

# directory = r'./noich_right_hemisphere_frame5_slice1_class_npydata/AD/patient004'
# max_val, min_val = -np.inf, np.inf
# img = np.load(file)
# data = img.get_fdata()
# if np.isnan(data).any():
#     print(f"文件 {file} 中包含 NaN 值，数据形状: {data.shape}")

# max_val = max(max_val, np.max(data))
# min_val = min(min_val, np.min(data))
# print(f"最小值: {min_val}, 最大值: {max_val}")

# # 遍历目录中的所有 .npy 文件
# for filename in os.listdir(directory):
#     if filename.endswith('.npy'):
#         file_path = os.path.join(directory, filename)
        
#         # 加载 .npy 文件
#         data = np.load(file_path)
        
#         # # 检查是否包含 NaN 值
#         # if np.isnan(data).any():
#         #     print(f"{filename} 包含 NaN 值")
#         # else:
#         #     print(f"{filename} 不包含 NaN 值")
            
#         # 打印文件名和最大值
#         print(data.shape)
#         max_value = np.max(data)
#         print(f"{filename} 的最大值: {max_value}")


npy_file = r'./noich_right_hemisphere_frame5_slice1_class_npydata/AD/patient004/slice_19.npy'
output_dir = r'./output_png'
output_nifti_dir = r'./output_nifti'
# 加载 .npy 文件
data = np.load(npy_file)

# 创建仿射矩阵（identity matrix）
# 将 NumPy 数据转换为 NIfTI 格式
os.makedirs(output_nifti_dir, exist_ok=True)
# 遍历第二个维度（120），将每个切片保存为单独的 NIfTI 文件
for i in range(data.shape[1]):  # 遍历 120 个切片
    slice_data = data[:, i, :, :]  # 提取单个切片，shape: (91, 46, 109)

    # 创建仿射矩阵（identity matrix）
    affine = np.eye(4)

    # 将切片数据转换为 NIfTI 格式
    nifti_img = nib.Nifti1Image(slice_data, affine)

    # 定义输出文件路径
    output_nifti_path = os.path.join(output_nifti_dir, f'slice_{str(i + 1).zfill(3)}.nii')

    # 保存为 NIfTI 文件
    nib.save(nifti_img, output_nifti_path)

    print(f"✅ 已保存切片 {i + 1} 到：{output_nifti_path}")

print("✅ 所有切片已保存完成！")
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历每一帧并保存为 PNG
for frame_idx in range(data.shape[1]):  # 遍历 16 帧
    frame_data = data[0, frame_idx, :, :]  # 提取单帧数据 (46, 109)
    
    # 保存为 PNG
    output_path = os.path.join(output_dir, f'slice_40_frame_{frame_idx + 1}.png')
    plt.figure(figsize=(6, 6))
    plt.imshow(frame_data, cmap='gray', origin='lower')
    plt.colorbar()
    plt.title(f'Slice 50 - Frame {frame_idx + 1}')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"PNG 图像已保存到 {output_dir}")

# 將每個 frame 存為臨時圖片後組成 GIF（或直接將每張影像寫入 imageio）
frames = []

for frame_idx in range(data.shape[1]):
    frame_data = data[0, frame_idx, :, :]  # shape: (46, 109)

    # 轉成 0-255 的 uint8 灰階影像
    norm_frame = (frame_data - np.min(frame_data)) / (np.max(frame_data) - np.min(frame_data) + 1e-8)
    gray_frame = (norm_frame * 255).astype(np.uint8)

    frames.append(gray_frame)

# 儲存為 GIF
imageio.mimsave(output_gif_path, frames, fps=3)  # 可調整 fps = 幀率

print(f"✅ GIF 已儲存：{output_gif_path}")