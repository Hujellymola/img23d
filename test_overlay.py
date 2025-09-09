import os
import cv2
import numpy as np
import matplotlib.cm as cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rgb_dir', type=str, required=True, help="Path to rendered RGB images")
parser.add_argument('--cluster_dir', type=str, required=True, help="Path to projected cluster images")
parser.add_argument('--output_path', type=str, required=True, help="Path to save overlay output")
args = parser.parse_args()

rgb_dir = os.path.expanduser(args.rgb_dir)
cluster_dir = os.path.expanduser(args.cluster_dir)
output_path = os.path.expanduser(args.output_path)

label_path = os.path.join(cluster_dir, "labels.npy")
labels = np.load(label_path)
indices = sorted(np.unique(labels).tolist())
# 提取所有index
# indices = [int(item["index"]) for item in data]
# indices = [0, 1, 2, 3, 4, 5, 6]  # 示例数据
# indices = [0, 1, 2, 3, 4]
# indices = [2, 2, 2, 2, 2, 2, 2]  # 示例数据
n_views = 8  

tab20 = cm.get_cmap("tab20", 20)
tab20_colors = (tab20(np.arange(20))[:, :3] * 255).astype(np.uint8)



# ==== 提取需要保留的RGB ====
# target_rgbs = [tab20_colors[index] for index in indices]
# print("Selected target RGBs:", target_rgbs)
for index in indices:
    target_rgbs = [tab20_colors[index]]
    print(f"Selected target RGB for index {index}: {target_rgbs}")
    output_path_ = os.path.join(output_path, f"index_{index}")

    for vid in range(n_views):
        # image_path = os.path.join(cluster_dir, f"pure_render_{vid:02d}.png")
        image_path = os.path.join(cluster_dir, f"pointcloud_proj_{vid:02d}.png")
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_rgb_path = os.path.join(rgb_dir, f"rgb_{vid:02d}.png")
        original_img = cv2.imread(original_rgb_path)

        # ==== 创建输出图（先全白） ====
        H, W, _ = img_rgb.shape
        output = np.ones_like(img_rgb, dtype=np.uint8) * 255

        # ==== 设置颜色容忍度 ====
        tolerance = 20

        mask = np.zeros((H, W), dtype=bool)  # 创建 mask
        # ==== 过滤逻辑 ====
        for target_rgb in target_rgbs:
            diff = np.linalg.norm(img_rgb - np.array(target_rgb), axis=2)
            mask_local = diff < tolerance
            mask |= mask_local  # 更新全局 mask
            # output[mask_local] = target_rgb  # 保留匹配颜色
            output[mask_local] = tab20_colors[2] # set all to the same color
            
        # cv2.imshow("Filtered Clusters", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        alpha = 0.5

        # === 生成 mask (非白色部分) ===
        # === 创建输出 ===
        overlay = original_img.copy()
        overlay[mask] = (original_img[mask] * (1 - alpha) + output[mask] * alpha).astype(np.uint8)

        # === 保存结果 ===
        os.makedirs(f"{output_path_}", exist_ok=True)
        cv2.imwrite(f"{output_path_}/overlay_{vid:02d}.png", overlay)
        print(f"✅ Saved blended overlay to {output_path_}/overlay_{vid:02d}.png")
