import argparse
import os
import json
import numpy as np
import open3d as o3d
import itertools
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# ===================== Globals formerly from visualize.py =====================
ROOT = None
OUTPUT_PATH = None
N_VIEWS = 8
H = None
W = None
FX = None
FY = None
CX = None
CY = None

# ====== Minimal reuse from visualize.py (unchanged logic) ======
def label_to_color(labels):
    """
    Map integer labels to RGB colors using tab20.
    """
    labels = np.asarray(labels).astype(int).ravel()
    from matplotlib import cm
    cmap = cm.get_cmap("tab20", 20)
    colors = cmap(labels % 20)[:, :3]
    return colors

def save_pointcloud_projection(points, colors, cam):
    """
    Project a colored point cloud into each view using the provided intrinsics/extrinsics
    and a naive nearest-neighbor color lookup at the backprojected depth point.
    Writes PNGs to OUTPUT_PATH. Requires globals:
      ROOT, OUTPUT_PATH, N_VIEWS, H, W, FX, FY, CX, CY
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    colors = np.asarray(colors)
    if colors.ndim == 1:
        # scalar labels -> map to RGB
        colors = label_to_color(colors.astype(int))
    if colors.shape[1] == 4:
        colors = colors[:, :3]
    assert colors.shape[1] == 3, f"colors must be Nx3, got {colors.shape}"

    kdtree = KDTree(points)
    for vid in range(N_VIEWS):
        # 初始化白底和深度缓冲
        proj_img = np.ones((H, W, 3), dtype=np.float32)
        depth_buffer = np.ones((H, W), dtype=np.float32) * 1e5

        # 加载深度图（与原逻辑保持一致）
        dep_path = os.path.join(ROOT, f"depth_{vid:02d}.npy")
        depth = np.load(dep_path)  # H×W

        # 相机外参（世界->相机）矩阵反转
        ext = np.array(cam["views"][f"{vid:02d}"])  # 世界到相机
        cam2world = np.linalg.inv(ext)

        for i in range(H):
            for j in range(W):
                z = depth[i, j]
                if z <= 1e-5 or z >= 10.0:
                    continue

                x = (j - CX) * z / FX
                y = (i - CY) * z / FY
                cam_point = np.array([x, -y, -z, 1.0])  # Blender 坐标系处理
                world_point = cam2world @ cam_point
                world_xyz = world_point[:3]

                # --- 最近点搜索 (可选: KDTree) ---
                dist, idx = kdtree.query(world_xyz)
                color = colors[idx]

                # --- 深度缓存 ---
                if z < depth_buffer[i, j]:
                    depth_buffer[i, j] = z
                    proj_img[i, j] = color

        # 保存图像
        out_path = f"{OUTPUT_PATH}/pointcloud_proj_{vid:02d}.png"
        plt.imsave(out_path, proj_img)
        print(f"✅ 已保存点云投影图像至 {out_path}")
        
# def save_pointcloud_projection(points, colors, cam):
#     colors = np.asarray(colors)
#     if colors.ndim == 1: colors = label_to_color(colors.astype(int))
#     if colors.shape[1] == 4: colors = colors[:, :3]
#     assert colors.shape[1] == 3

#     kdtree = KDTree(points)
#     for vid in range(N_VIEWS):
#         proj_img = np.ones((H, W, 3), dtype=np.float32)
#         depth_buffer = np.ones((H, W), dtype=np.float32) * 1e9
        
#         depth = np.load(os.path.join(ROOT, f"depth_{vid:02d}.npy"))
#         ext = np.array(cam["views"][f"{vid:02d}"])          # world->cam
#         cam2world = np.linalg.inv(ext)
#         R = cam2world[:3, :3]
#         t = cam2world[:3, 3]

#         for i in range(H):
#             yi = (i - CY) / FY
#             for j in range(W):
#                 d = float(depth[i, j])
#                 if not (0.1 <= d <= 10.0):  # your DEP_RANGE
#                     continue
#                 xi = (j - CX) / FX
#                 # camera-space ray (Blender: -Z forward, Y up)
#                 dir_cam = np.array([xi, -yi, -1.0], dtype=np.float32)
#                 dir_cam /= np.linalg.norm(dir_cam) + 1e-12
#                 world_xyz = t + d * (R @ dir_cam)

#                 _, idx = kdtree.query(world_xyz)
#                 color = colors[idx]
#                 if d < depth_buffer[i, j]:
#                     depth_buffer[i, j] = d
#                     proj_img[i, j] = color
#                 # proj_img[i, j] = colors[idx]

#         out_path = f"{OUTPUT_PATH}/pointcloud_proj_{vid:02d}.png"
#         plt.imsave(out_path, np.clip(proj_img, 0, 1))
#         print(f"saved {out_path}")

# ------------------------------------------------------------------------------
# 自动对齐点云到 Blender 世界坐标系
# ------------------------------------------------------------------------------

def backproject_world_points_all(cam, FX,FY,CX,CY, step=8, dep_range=(0.1,10.0)):
    P = []
    for vid in range(N_VIEWS):
        dep = np.load(os.path.join(ROOT, f"depth_{vid:02d}.npy"))
        H,W = dep.shape
        ext = np.array(cam["views"][f"{vid:02d}"])
        cam2world = np.linalg.inv(ext); R = cam2world[:3,:3]; t = cam2world[:3,3]
        for i in range(0,H,step):
            yi = (i - CY) / FY
            for j in range(0,W,step):
                d = float(dep[i,j])
                if dep_range[0] <= d <= dep_range[1]:
                    xi = (j - CX) / FX
                    dir_cam = np.array([xi, -yi, -1.0], np.float32)
                    dir_cam /= np.linalg.norm(dir_cam) + 1e-12
                    P.append(t + d * (R @ dir_cam))
    return np.asarray(P)

def center_and_radius(X):
    c = X.mean(0)
    r = np.median(np.linalg.norm(X - c, axis=1))
    return c, max(r, 1e-9)

def fix_det(R):
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R

def pca_axes(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return fix_det(Vt.T)  # 3x3

def enum_axis_rotations():
    mats = []
    basis = np.eye(3)
    for perm in itertools.permutations([0,1,2]):
        Pperm = basis[:, list(perm)]
        for sx,sy,sz in itertools.product([1,-1],[1,-1],[1,-1]):
            S = np.diag([sx,sy,sz])
            R = Pperm @ S
            if np.isclose(np.linalg.det(R), 1.0):  # 保持右手系
                mats.append(R)
    # 去重
    uniq = []
    for R in mats:
        if not any(np.allclose(R, U) for U in uniq):
            uniq.append(R)
    return uniq

def median_nn_after_transform(points, P_world, R, s, t):
    kdt = KDTree(P_world)
    Q = (R @ points.T).T * s + t
    dists, _ = kdt.query(Q, k=1)
    return np.median(dists)

def auto_align_points_to_blender(points, cam, FX,FY,CX,CY):
    # 1) 构建多视角世界点
    P = backproject_world_points_all(cam, FX,FY,CX,CY, step=8)
    cP, rP = center_and_radius(P)
    cQ, rQ = center_and_radius(points)

    # 2) PCA 初值
    VP = pca_axes(P)
    VQ = pca_axes(points)
    R_pca = fix_det(VP @ VQ.T)
    s_pca = rP / rQ
    t_pca = cP - (R_pca @ cQ) * s_pca
    best = (median_nn_after_transform(points, P, R_pca, s_pca, t_pca), R_pca, s_pca, t_pca)

    # 3) 离散枚举微调
    for Rc in enum_axis_rotations():
        s = rP / rQ
        t = cP - (Rc @ cQ) * s
        m = median_nn_after_transform(points, P, Rc, s, t)
        if m < best[0]:
            best = (m, Rc, s, t)

    m,R,s,t = best
    return m, R, s, t, P
    
def normalize_like_blender_unit_sphere(points):
    # bbox center
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    # bbox diagonal -> bounding-sphere radius
    diag = np.linalg.norm(maxs - mins)
    r_curr = max(diag * 0.5, 1e-6)
    s = 1.0 / r_curr  # Blender used target_r = 1.0
    return (points - center) * s

# estimate the world center from depths (fast subsample)
def estimate_world_center(root, cam, FX, FY, CX, CY, step=8):
    dep = np.load(os.path.join(root, "depth_00.npy"))
    H0, W0 = dep.shape
    ext = np.array(cam["views"]["00"])
    cam2world = np.linalg.inv(ext)
    R, t = cam2world[:3,:3], cam2world[:3,3]
    pts = []
    for i in range(0, H0, step):
        yi = (i - CY) / FY
        for j in range(0, W0, step):
            d = float(dep[i, j])
            if 0.1 <= d <= 10.0:
                xi = (j - CX) / FX
                ray = np.array([xi, -yi, -1.0], dtype=np.float32)
                ray /= np.linalg.norm(ray) + 1e-12
                pts.append(t + d * (R @ ray))
    return np.mean(np.stack(pts, 0), 0)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, help="Folder with rgb_XX.png, depth_XX.npy, camera.json")
    p.add_argument('--output', type=str, required=True, help="Output dir for projections")
    p.add_argument('--points_ply', type=str, required=True, help="Path to point cloud .ply")
    p.add_argument('--clustering_npy', type=str, required=True, help="Path to labels .npy")
    p.add_argument('--n_views', type=int, default=8)
    return p.parse_args()

#################################### Main function ####################################
def main():
    global ROOT, OUTPUT_PATH, N_VIEWS, H, W, FX, FY, CX, CY

    args = parse_arguments()

    # Set globals used by save_pointcloud_projection
    ROOT = os.path.expanduser(args.input)
    OUTPUT_PATH = os.path.expanduser(args.output)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    cam_json = os.path.join(ROOT, "camera.json")
    with open(cam_json, 'r') as f:
        cam = json.load(f)

    intr = np.array(cam["intrinsic"])
    FX, _, CX = intr[0]
    _, FY, CY = intr[1]
    N_VIEWS = int(args.n_views)

    # Infer H, W from any depth
    dep0 = np.load(os.path.join(ROOT, "depth_00.npy"))
    H, W = dep0.shape

    # Load points and labels
    pcd = o3d.io.read_point_cloud(os.path.expanduser(args.points_ply))
    points = np.asarray(pcd.points)  # [N,3]
        
    labels = np.load(os.path.expanduser(args.clustering_npy)).astype(int).ravel()  # [N]
    assert len(labels) == len(points), f"labels {len(labels)} != points {len(points)}"
    
    # Colors from labels
    colors = label_to_color(labels)  # [N,3] in [0,1]
    
    points = normalize_like_blender_unit_sphere(points)  # Normalize to unit sphere
    world_center = estimate_world_center(ROOT, cam, FX, FY, CX, CY)
    points = points + world_center  # move KDTree into Blender world 
    
    # 读入 points 与 labels 后：
    m0, R, s, t, P = auto_align_points_to_blender(points, cam, FX, FY, CX, CY)
    print(f"[auto-align] median NN after transform: {m0:.4f}")
    points = (R @ points.T).T * s + t  # 应用刚体+尺度
 
    # ######################################## DEBUG ########################################
    # def backproj_variant(i, j, d, FX,FY,CX,CY, cam2world, variant):
    #     R = cam2world[:3,:3]; t = cam2world[:3,3]
    #     xi = (j - CX) / FX
    #     yi = (i - CY) / FY
    #     if variant == "ray":            # 视距模型
    #         dir_cam = np.array([xi, -yi, -1.0], np.float32)
    #         dir_cam /= np.linalg.norm(dir_cam) + 1e-12
    #         return t + d * (R @ dir_cam)
    #     elif variant == "pinhole":      # 光轴深度模型
    #         x = xi * d; y = yi * d
    #         cam_h = np.array([x, -y, -d, 1.0])
    #         return (cam2world @ cam_h)[:3]
    #     else:
    #         raise ValueError

    # def median_nn_for_variant(variant):
    #     dep = np.load(os.path.join(ROOT, "depth_00.npy"))
    #     H0,W0 = dep.shape
    #     ext = np.array(cam["views"]["00"]); cam2world = np.linalg.inv(ext)
    #     kdt = KDTree(points)
    #     ds=[]
    #     for i in range(0,H0,16):
    #         for j in range(0,W0,16):
    #             d = float(dep[i,j])
    #             if 0.1 <= d <= 10.0:
    #                 p = backproj_variant(i,j,d, FX,FY,CX,CY, cam2world, variant)
    #                 dist,_ = kdt.query(p)
    #                 ds.append(dist)
    #     return np.median(ds)

    # print("median NN — ray :", median_nn_for_variant("ray"))
    # print("median NN — pin :", median_nn_for_variant("pinhole"))
    # ##################################################################################################

    # Project
    save_pointcloud_projection(points, colors, cam)

if __name__ == '__main__':
    main()
