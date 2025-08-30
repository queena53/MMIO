import os
import shutil
from glob import glob

def collect_all_gif(input_root, output_dir, class_name="AD"):
    """
    將所有 patientXXX/slice_XX/layer_name/*.gif 檔案複製到 output_dir，
    並將檔名加上 patientXXX_sliceXX_layername 前綴。
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(input_root, class_name, "patient*", "slice_*", "*", "*_animation.gif")
    all_gif_paths = glob(pattern)

    print(f"🔍 找到 {len(all_gif_paths)} 個 .gif 檔案")

    for gif_path in all_gif_paths:
        parts = gif_path.split(os.sep)
        try:
            patient = parts[-4]       # patientXXX
            slice_name = parts[-3]    # slice_XX
            layer_name = parts[-2]    # layer_name
            new_filename = f"{patient}_{slice_name}_{layer_name}_animation.gif"
            shutil.copy(gif_path, os.path.join(output_dir, new_filename))
        except Exception as e:
            print(f"❌ 錯誤處理 {gif_path}: {e}")

    print(f"✅ 所有 gif 檔案已複製到 {output_dir}")

if __name__ == "__main__":
    # 修改這裡的 class_name 可處理 CAA
    collect_all_gif(
        input_root="gradcam_outputs/mask_frame22_24_lr2",
        output_dir="gradcam_outputs/mask_frame22_24_lr2_gif_all/CAA_ICH",
        class_name="CAA_ICH"
    )
