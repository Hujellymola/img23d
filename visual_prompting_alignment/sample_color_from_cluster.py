import argparse
import os
import json
import numpy as np
import open3d as o3d
import visualize as viz
import itertools
from scipy.spatial import KDTree

def backproject_world_points_all(cam, FX,FY,CX,CY, step=8, dep_range=(0.1,10.0)):
    P = []
    for vid in range(viz.N_VIEWS):
        dep = np.load(os.path.join(viz.ROOT, f"depth_{vid:02d}.npy"))
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
        P = basis[:, list(perm)]
        for sx,sy,sz in itertools.product([1,-1],[1,-1],[1,-1]):
            S = np.diag([sx,sy,sz])
            R = P @ S
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
    H, W = dep.shape
    ext = np.array(cam["views"]["00"])
    cam2world = np.linalg.inv(ext)
    R, t = cam2world[:3,:3], cam2world[:3,3]
    pts = []
    for i in range(0, H, step):
        yi = (i - CY) / FY
        for j in range(0, W, step):
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
    args = parse_arguments()

    # Set globals used by visualize.save_pointcloud_projection
    viz.ROOT = os.path.expanduser(args.input)
    viz.OUTPUT_PATH = os.path.expanduser(args.output)
    os.makedirs(viz.OUTPUT_PATH, exist_ok=True)

    cam_json = os.path.join(viz.ROOT, "camera.json")
    with open(cam_json, 'r') as f:
        cam = json.load(f)

    intr = np.array(cam["intrinsic"])
    viz.FX, _, viz.CX = intr[0]
    _, viz.FY, viz.CY = intr[1]
    viz.N_VIEWS = int(args.n_views)

    # Infer H, W from any depth
    dep0 = np.load(os.path.join(viz.ROOT, "depth_00.npy"))
    viz.H, viz.W = dep0.shape

    # Load points and labels
    pcd = o3d.io.read_point_cloud(os.path.expanduser(args.points_ply))
    points = np.asarray(pcd.points)  # [N,3]
        
    labels = np.load(os.path.expanduser(args.clustering_npy)).astype(int).ravel()  # [N]
    assert len(labels) == len(points), f"labels {len(labels)} != points {len(points)}"
    
    # Colors from labels
    colors = viz.label_to_color(labels)  # [N,3] in [0,1]
    
    points= normalize_like_blender_unit_sphere(points)  # Normalize to unit sphere
    world_center = estimate_world_center(viz.ROOT, cam, viz.FX, viz.FY, viz.CX, viz.CY)
    points = points + world_center  # move KDTree into Blender world 
    
    # 读入 points 与 labels 后：
    m0, R, s, t, P = auto_align_points_to_blender(points, cam, viz.FX,viz.FY,viz.CX,viz.CY)
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
    #         cam = np.array([x, -y, -d, 1.0])
    #         return (cam2world @ cam)[:3]
    #     else:
    #         raise ValueError

    # def median_nn_for_variant(variant):
    #     dep = np.load(os.path.join(viz.ROOT, "depth_00.npy"))
    #     H,W = dep.shape
    #     ext = np.array(cam["views"]["00"]); cam2world = np.linalg.inv(ext)
    #     ds=[]
    #     for i in range(0,H,16):
    #         for j in range(0,W,16):
    #             d = float(dep[i,j])
    #             if 0.1 <= d <= 10.0:
    #                 p = backproj_variant(i,j,d, viz.FX,viz.FY,viz.CX,viz.CY, cam2world, variant)
    #                 dist,_ = kdt.query(p)
    #                 ds.append(dist)
    #     return np.median(ds)

    # print("median NN — ray :", median_nn_for_variant("ray"))
    # print("median NN — pin :", median_nn_for_variant("pinhole"))
    # ##################################################################################################

    # Project
    viz.save_pointcloud_projection(points, colors, cam)

if __name__ == '__main__':
    main()
