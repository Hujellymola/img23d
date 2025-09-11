import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import einops
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import hdbscan
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from scipy.spatial import KDTree
from matplotlib import cm
import argparse

def save_voxel_index_mask(voxel_centers, voxel_colors, cam, min_bound, voxel_size):
    # 假设第一张 depth 拿来初始化shape
    depth0 = np.load(os.path.join(ROOT, f"depth_00.npy"))
    H, W = depth0.shape

    # === 创建 voxel_idx -> label 映射 ===
    voxel_index_dict = {}
    for center, idx in zip(voxel_centers, voxel_colors):
        voxel_idx = tuple(np.floor((center - min_bound) / voxel_size).astype(int))
        voxel_index_dict[voxel_idx] = idx


    for vid in range(N_VIEWS):

        proj_img = np.ones((H, W, 3), dtype=np.float32)

        # 读取深度
        dep_path = os.path.join(ROOT, f"depth_{vid:02d}.npy")
        depth = np.load(dep_path)

        ext = np.array(cam["views"][f"{vid:02d}"])
        cam2world = np.linalg.inv(ext)

        for i in range(H):
            for j in range(W):
                z = depth[i, j]
                if z <= 1e-5 or z >= 10.0:
                    continue

                # Step1: 从像素反投影
                x = (j - CX) * z / FX
                y = (i - CY) * z / FY
                cam_point = np.array([x, -y, -z, 1.0])

                world_point = cam2world @ cam_point
                world_xyz = world_point[:3]

                voxel_idx = tuple(np.floor((world_xyz - min_bound) / voxel_size).astype(int))

                if voxel_idx in voxel_index_dict:
                    proj_img[i, j] = voxel_index_dict[voxel_idx]

        # # 可视化
        # plt.figure(figsize=(6,6))
        # plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
        # plt.title(f"Voxel Index Mask (View {vid:02d})")
        # plt.axis("off")
        # plt.show()

        # 保存npy
        os.makedirs(f"{OUTPUT_PATH}", exist_ok=True)
        plt.imsave(f"{OUTPUT_PATH}/proj_{vid:02d}.png", proj_img)
        print(f"✅ 已保存 index mask 至 {OUTPUT_PATH}/proj_{vid:02d}.png")

# ====== save voxel-based 2d image ======
def save_voxel_based_image(voxel_centers, voxel_features, depth, cam, min_bound, voxel_size):
    H, W = depth.shape
    proj_img = np.ones((H, W, 3), dtype=np.float32)  # 全白底

    voxel_color_dict = {}

    for center, color in zip(voxel_centers, voxel_features):
        voxel_idx = tuple(np.floor((center - min_bound) / voxel_size).astype(int))
        voxel_color_dict[voxel_idx] = color  # 可以是 RGB 或 PCA 映射到 RGB 的值

    for vid in range(N_VIEWS):
        proj_img = np.ones((H, W, 3), dtype=np.float32)  # 全白底
        for i in range(H):
            for j in range(W):
                dep_path = os.path.join(ROOT, f"depth_{vid:02d}.npy")
                depth = np.load(dep_path)  # H×W
                z = depth[i, j]
                if z <= 1e-5 or z >= 10.0:  # 可调整深度范围
                    continue

                # Step 1: 从像素反投影到相机坐标
                x = (j - CX) * z / FX
                y = (i - CY) * z / FY
                cam_point = np.array([x, -y, -z, 1.0])  # 注意 Blender 的坐标设置

                ext = np.array(cam["views"][f"{vid:02d}"])  # 世界到相机
                cam2world = np.linalg.inv(ext)

                # Step 2: 转换为世界坐标
                world_point = cam2world @ cam_point
                world_xyz = world_point[:3]

                # Step 3: 找到所在 voxel 的索引
                voxel_idx = tuple(np.floor((world_xyz - min_bound) / voxel_size).astype(int))

                # Step 4: 查找该 voxel 的颜色（如果有）
                if voxel_idx in voxel_color_dict:
                    proj_img[i, j] = voxel_color_dict[voxel_idx]

        # 保存图像
        os.makedirs("result", exist_ok=True)
        plt.imsave(f"result/_dense_voxel_proj_{vid:02d}.png", proj_img)
        print(f"✅ 已保存致密 2D 投影图像至 result/dense_voxel_proj{vid:02d}.png")

# def save_pointcloud_projection(points, colors, cam):
#     os.makedirs("result", exist_ok=True)
#     colors = np.asarray(colors)
#     if colors.ndim == 1:
#         # scalar labels -> map to RGB
#         colors = label_to_color(colors.astype(int))
#     if colors.shape[1] == 4:
#         colors = colors[:, :3]
#     assert colors.shape[1] == 3, f"colors must be Nx3, got {colors.shape}"
#     kdtree = KDTree(points)
#     for vid in range(N_VIEWS):
#         # 初始化白底和深度缓冲
#         proj_img = np.ones((H, W, 3), dtype=np.float32)
#         depth_buffer = np.ones((H, W), dtype=np.float32) * 1e5

#         # 加载深度图（与原逻辑保持一致）
#         dep_path = os.path.join(ROOT, f"depth_{vid:02d}.npy")
#         depth = np.load(dep_path)  # H×W

#         # 相机外参（世界->相机）矩阵反转
#         ext = np.array(cam["views"][f"{vid:02d}"])  # 世界到相机
#         cam2world = np.linalg.inv(ext)

#         for i in range(H):
#             for j in range(W):
#                 z = depth[i, j]
#                 if z <= 1e-5 or z >= 10.0:
#                     continue

#                 x = (j - CX) * z / FX
#                 y = (i - CY) * z / FY
#                 cam_point = np.array([x, -y, -z, 1.0])  # Blender 坐标系处理
#                 world_point = cam2world @ cam_point
#                 world_xyz = world_point[:3]

#                 # --- 最近点搜索 (可选: KDTree) ---
#                 # 这里用最近邻找颜色（稠密点云可以近似）
#                 dist, idx = kdtree.query(world_xyz)
#                 color = colors[idx]

#                 # --- 深度缓存 ---
#                 if z < depth_buffer[i, j]:
#                     depth_buffer[i, j] = z
#                     proj_img[i, j] = color

#         # 保存图像
#         out_path = f"{OUTPUT_PATH}/pointcloud_proj_{vid:02d}.png"
#         plt.imsave(out_path, proj_img)
#         print(f"✅ 已保存点云投影图像至 {out_path}")

def save_pointcloud_projection(points, colors, cam):
    colors = np.asarray(colors)
    if colors.ndim == 1: colors = label_to_color(colors.astype(int))
    if colors.shape[1] == 4: colors = colors[:, :3]
    assert colors.shape[1] == 3

    kdtree = KDTree(points)
    for vid in range(N_VIEWS):
        proj_img = np.ones((H, W, 3), dtype=np.float32)
        depth_buffer = np.ones((H, W), dtype=np.float32) * 1e9
        
        depth = np.load(os.path.join(ROOT, f"depth_{vid:02d}.npy"))
        ext = np.array(cam["views"][f"{vid:02d}"])          # world->cam
        cam2world = np.linalg.inv(ext)
        R = cam2world[:3, :3]
        t = cam2world[:3, 3]

        for i in range(H):
            yi = (i - CY) / FY
            for j in range(W):
                d = float(depth[i, j])
                if not (0.1 <= d <= 10.0):  # your DEP_RANGE
                    continue
                xi = (j - CX) / FX
                # camera-space ray (Blender: -Z forward, Y up)
                dir_cam = np.array([xi, -yi, -1.0], dtype=np.float32)
                dir_cam /= np.linalg.norm(dir_cam) + 1e-12
                world_xyz = t + d * (R @ dir_cam)

                _, idx = kdtree.query(world_xyz)
                color = colors[idx]
                if d < depth_buffer[i, j]:
                    depth_buffer[i, j] = d
                    proj_img[i, j] = color
                # proj_img[i, j] = colors[idx]

        out_path = f"{OUTPUT_PATH}/pointcloud_proj_{vid:02d}.png"
        plt.imsave(out_path, np.clip(proj_img, 0, 1))
        print(f"saved {out_path}")

def render_voxel_mesh(voxel_mesh, cam, name="test_proj"):
        # === 一次性初始化渲染器与场景 ===
    W, H = 448, 448
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    scene = renderer.scene

    # 材质设置
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    # material.base_color = [1, 1, 1, 1]

    scene.add_geometry("voxel_mesh", voxel_mesh, material)

    # 获取相机内参
    intrinsic_mat = np.array(cam["intrinsic"])
    fx, _, cx = intrinsic_mat[0]
    _, fy, cy = intrinsic_mat[1]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)
    K = intrinsic.intrinsic_matrix

    for vid in range(N_VIEWS):
        extrinsic = np.array(cam["views"][f"{vid:02d}"])  # 世界→相机
        cam2world = np.linalg.inv(extrinsic)  # 相机→世界

        # === 提取 Blender 相机的位姿信息 ===
        cam_position = cam2world[:3, 3]                    # 相机位置
        cam_forward = -cam2world[:3, 2]                     # 相机 Z 轴方向（朝前）
        cam_up = cam2world[:3, 1]                         # 相机 Y 轴方向（朝上），注意要取反以匹配 Open3D

        cam_target = cam_position + cam_forward * 1.0      # 视点朝向


        # === 设置摄像机位置和朝向 ===
        scene.camera.look_at(cam_target, cam_position, cam_up)
        scene.camera.set_projection(K, 0.001, 10.0, W, H)
        print("Camera position (from extrinsic):", np.linalg.inv(extrinsic)[:3, 3])
        print("Object center:", voxel_mesh.get_axis_aligned_bounding_box().get_center())


        # === 渲染并保存图像 ===
        img = renderer.render_to_image()
        os.makedirs(f"{OUTPUT_PATH}", exist_ok=True)
        o3d.io.write_image(f"{OUTPUT_PATH}/{name}_{vid:02d}.png", img)
        print(f"✅ 渲染完成，图像保存至 {OUTPUT_PATH}/{name}_{vid:02d}.png")
    
def create_camera_geometry(cam2world, scale=0.05):
    # 相机坐标系的三个轴 (X=红, Y=绿, Z=蓝)
    origin = cam2world[:3, 3]

    x_axis = cam2world[:3, 0] * scale
    y_axis = cam2world[:3, 1] * scale
    z_axis = cam2world[:3, 2] * scale

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scale, origin=[0, 0, 0]
    )
    mesh_frame.translate(origin)
    mesh_frame.rotate(cam2world[:3, :3], center=origin)

    return mesh_frame

# clustering and create voxel mesh
def cluster_voxels(voxel_centers, voxel_features, voxel_size=0.03):
    # mean_shift = MeanShift()
    # labels = mean_shift.fit_predict(voxel_features)
    # print(f"MeanShift found {len(set(labels))} clusters.")

    # # 若 cluster 数太少（如 < 5），则使用 KMeans
    # if len(set(labels)) < 5:
    #     kmeans = KMeans(n_clusters=7, random_state=0)
    #     labels = kmeans.fit_predict(voxel_features)
    kmeans = KMeans(n_clusters=7, random_state=0)
    labels = kmeans.fit_predict(voxel_features)

    colors_ = label_to_color(labels)

    assert len(colors_) == len(voxel_centers), "Colors length mismatch!"
    all_meshes = []
    for i in range(len(colors_)):
        # 每个 voxel 的中心点
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size,
                                                    height=voxel_size,
                                                    depth=voxel_size)
        cube.translate(voxel_centers[i] - voxel_size / 2)
        cube.paint_uniform_color(np.clip(colors_[i], 0, 1))

        all_meshes.append(cube)

    # 合并所有 box 成一个大 mesh
    voxel_mesh = all_meshes[0]
    for mesh in all_meshes[1:]:
        voxel_mesh += mesh
    return voxel_mesh, voxel_centers, colors_

# 为每个 cluster 赋一个颜色
def label_to_color(labels):
    # cmap = cm.get_cmap("tab20", np.max(labels) + 1)
    # colors = cmap(labels % cmap.N)[:, :3]
    # return colors
    labels = np.asarray(labels).astype(int).ravel()  # <-- flatten to (N,)
    cmap = cm.get_cmap("tab20", 20)  # 固定20色
    colors = cmap(labels % 20)[:, :3]
    return colors

def create_voxel_mesh(all_pts, colors, voxel_size=0.03):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)

    # 获取 AABB
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound

    # 设置 voxel 大小
    voxel_size = voxel_size  # 可以调小调大

    # 计算 voxel 网格坐标
    voxel_indices = np.floor((all_pts - min_bound) / voxel_size).astype(int)
    voxel_dict = defaultdict(list)

    # 将每个点加入对应 voxel
    for i, voxel in enumerate(voxel_indices):
        key = tuple(voxel)
        voxel_dict[key].append(i)

    # # 构建 voxel 中心位置和融合特征
    # voxel_centers = []
    # voxel_features = []

    # for key, idx_list in voxel_dict.items():
    #     pts_in_voxel = all_pts[idx_list]
    #     # feats_in_voxel = colors[idx_list]

    #     center = min_bound + (np.array(key) + 0.5) * voxel_size
    #     avg_feat = np.mean(colors[idx_list], axis=0)

    #     voxel_centers.append(center)
    #     voxel_features.append(avg_feat)

    # voxel_centers = np.array(voxel_centers)     # shape: (M, 3)
    # voxel_features = np.array(voxel_features)   # shape: (M, C)

    all_meshes = []

    for voxel_idx, indices in voxel_dict.items():
        # 每个 voxel 的中心点
        center = min_bound + (np.array(voxel_idx) + 0.5) * voxel_size
        color = np.mean(colors[indices], axis=0)

        # 构建一个单位 cube，然后缩放 & 平移到中心位置
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size,
                                                    height=voxel_size,
                                                    depth=voxel_size)
        cube.translate(center - voxel_size / 2)
        cube.paint_uniform_color(np.clip(color, 0, 1))

        all_meshes.append(cube)

    # 合并所有 box 成一个大 mesh
    voxel_mesh = all_meshes[0]
    for mesh in all_meshes[1:]:
        voxel_mesh += mesh
    # voxel_mesh.compute_vertex_normals()
    return voxel_mesh


def farthest_point_sample(xyz: torch.Tensor, K: int) -> torch.Tensor:
    """
    从 xyz (N,3) 中最远点采样出 K 个点的索引(FPS)
    如果 N <= K, 直接返回 all indices。
    """
    N, _ = xyz.shape
    if N <= K:
        return torch.arange(N, device=xyz.device)
    centroids = torch.zeros(K, dtype=torch.long, device=xyz.device)
    distances = torch.full((N,), float('inf'), device=xyz.device)
    farthest = torch.randint(0, N, (1,), device=xyz.device).item()
    for i in range(K):
        centroids[i] = farthest
        diff = xyz - xyz[farthest:farthest+1]       # (N,3)
        dist = (diff*diff).sum(dim=1)               # (N,)
        distances = torch.minimum(distances, dist)  
        farthest = torch.argmax(distances).item()
    return centroids  # (K,)






if __name__ == "__main__":
    import argparse, json, os, numpy as np, torch, einops, torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # ====== 参数设置 ======
    # ROOT = os.path.expanduser("~/UAD/output/demo_views_2")
    ROOT  = os.path.expanduser(args.input)
    OUTPUT_PATH = os.path.expanduser(args.output)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # OUTPUT_PATH = os.path.expanduser("~/UAD/result/test_cluster")
    # ROOT = os.path.expanduser("~/UAD/output/test_render")
    CAM_JSON  = os.path.join(ROOT, "camera.json")
    N_VIEWS = 8
    DEP_RANGE = (0.1, 10)
    with open(CAM_JSON, 'r') as f:
        cam = json.load(f)
    intr = np.array(cam["intrinsic"])
    FX, _, CX = intr[0]
    _, FY, CY = intr[1]
    IMG_SIZE = 448
    FEAT_SIZE = 32  # 对应 ViT patch 数量

    # ====== DINOv2 模型加载 ======
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    dino.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # ====== 加载相机参数 ======
    with open(os.path.join(ROOT, "camera.json")) as f:
        cam = json.load(f)

    all_pts, all_feats = [], []
    all_cols = []  

    for vid in range(N_VIEWS):
        # === 加载图像与深度 ===
        rgb_path = os.path.join(ROOT, f"rgb_{vid:02d}.png")
        dep_path = os.path.join(ROOT, f"depth_{vid:02d}.npy")
        ext = np.array(cam["views"][f"{vid:02d}"])  # 世界到相机
        cam2world = np.linalg.inv(ext)

        rgb = Image.open(rgb_path).convert("RGB")
        col = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
        col = col[..., :3]
        depth = np.load(dep_path)

        # === DINO 特征提取 ===
        img_tensor = transforms.ToTensor()(rgb)
        img_tensor = normalize(img_tensor).unsqueeze(0)
        with torch.no_grad():
            features_dict = dino.forward_features(img_tensor)
            patch_tokens = features_dict['x_norm_patchtokens']  # [1, 1024, 768]
            feat_tokens = einops.rearrange(patch_tokens, 'b (h w) c -> b c h w', h=FEAT_SIZE)
            feat_up = F.interpolate(feat_tokens, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            feat_flat = einops.rearrange(feat_up, 'b c h w -> b (h w) c')

        # === 点云重建 ===
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        ones = np.ones_like(z)
        pts_cam = np.stack([x, -y, -z, ones], axis=-1)  # 注意这里必须是 -y, -z

        mask = (z > DEP_RANGE[0]) & (z < DEP_RANGE[1])
        pts_world = (cam2world @ pts_cam.reshape(-1, 4).T).T[:, :3]
        pts_world = pts_world[mask.ravel()]
        feats = feat_flat.squeeze(0).cpu().numpy()[mask.ravel()]

        all_pts.append(pts_world)
        all_feats.append(feats)

        # 颜色
        cols = col[v, u].reshape(-1,3)
        cols = cols[mask.ravel()]
        all_cols.append(cols)


    # ====== 拼接 & PCA 降维颜色可视化 ======
    all_pts = np.concatenate(all_pts, axis=0)
    all_feats = np.concatenate(all_feats, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)

    # === 2. FPS on filtered_pts ===
    xyz = torch.from_numpy(all_pts).float().cuda()  # shape: [M, 3]
    K = 20000  # or any number of final points you want
    fps_idx = farthest_point_sample(xyz, K).cpu().numpy()

    # === 3. 得到最终点云和特征 ===
    # final_pts = all_pts[fps_idx]       # [K, 3]
    # final_feats = all_feats[fps_idx]   # [K, C]

    final_pts = all_pts  # [N, 3]
    final_feats = all_feats  # [N, C]

    pca = PCA(n_components=3)
    feats_pca = pca.fit_transform(final_feats)
    colors = (feats_pca - feats_pca.min()) / (feats_pca.max() - feats_pca.min())

    pca_16 = PCA(n_components=16)
    feats_pca_16 = pca_16.fit_transform(final_feats)
    # ===== 平滑颜色 ======
    radius = 0.02  # 半径范围，单位与点云单位一致

    # 构造 point cloud 和 KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    # pcd.colors = o3d.utility.Vector3dVector(colors)  # 如果 colors 是 384 维的，用前3维临时占位也可
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 初始化平滑后的数组
    smoothed_features = np.zeros_like(feats_pca_16)

    for i in range(len(all_pts)):
        [_, idxs, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        smoothed_features[i] = np.mean(feats_pca_16[idxs], axis=0)

    X = smoothed_features

    # ========== Step 1: 用子采样估计带宽 ==========
    # subset_ratio = 0.01  # 1% 子采样
    # subset_idx = np.random.choice(X.shape[0], int(subset_ratio * X.shape[0]), replace=False)
    # subset = X[subset_idx]
    # bandwidth = estimate_bandwidth(subset, quantile=0.2)
    # print(f"Estimated bandwidth: {bandwidth}")
    # bandwidth = bandwidth * 0.3

    # # # ========== Step 2: 用子采样聚类 ==========
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    # labels = ms.fit_predict(smoothed_features)
    # print(f"MeanShift found {len(set(labels))} clusters.")

    # mean_shift = MeanShift()
    # labels = mean_shift.fit_predict(smoothed_features)
    # print(f"MeanShift found {len(set(labels))} clusters.")

    # 若 cluster 数太少（如 < 5），则使用 KMeans
    # if len(set(labels)) < 5:
    #     kmeans = KMeans(n_clusters=6, random_state=0)
    #     labels = kmeans.fit_predict(smoothed_features)
    feat_scaler = StandardScaler(with_mean=True, with_std=True)
    xyz_scaler  = StandardScaler(with_mean=True, with_std=True)

    # F = feat_scaler.fit_transform(smoothed_features)        # (N, 16)
    # X = xyz_scaler.fit_transform(final_pts)        # (N, 3)

    # --- 3) Balance feature vs. spatial influence ---
    # alpha controls feature weight, beta controls spatial weight
    alpha = 1.0      # try 1.0, then tune
    beta  = 0.3      # try 0.25–1.0; increase for more spatial smoothness
    num_clu = 4

    # Z = np.hstack([alpha * F, beta * X])           # (N, 19)
    Z = np.hstack([smoothed_features, beta * final_pts])  # (N, 19)
    # --- 4) Cluster ---
    kmeans = KMeans(n_clusters=num_clu, random_state=0)
    labels = kmeans.fit_predict(Z)
    # kmeans = KMeans(n_clusters=5, random_state=0)
    # labels = kmeans.fit_predict(smoothed_features)
    color_16 = label_to_color(labels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.colors = o3d.utility.Vector3dVector(color_16)  
    o3d.visualization.draw_geometries([pcd], window_name="DINO Feature-Fused PointCloud")

    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(final_pts)
    o3d.io.write_point_cloud(f"{OUTPUT_PATH}/points.ply", pcd_)
    np.save(f"{OUTPUT_PATH}/labels.npy", labels)
    with open(CAM_JSON, 'r') as f:
        cam = json.load(f)
    save_pointcloud_projection(final_pts, color_16, cam)
