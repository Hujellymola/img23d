import argparse
import os
import json
import numpy as np
import open3d as o3d

import visualize as viz  # import module so we can set its globals

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, help="Folder with rgb_XX.png, depth_XX.npy, camera.json")
    p.add_argument('--output', type=str, required=True, help="Output dir for projections")
    p.add_argument('--points_ply', type=str, required=True, help="Path to point cloud .ply")
    p.add_argument('--clustering_npy', type=str, required=True, help="Path to labels .npy")
    p.add_argument('--n_views', type=int, default=8)
    return p.parse_args()

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

    labels = np.load(os.path.expanduser(args.clustering_npy))  # [N]
    assert len(labels) == len(points), f"labels {len(labels)} != points {len(points)}"

    # Colors from labels
    colors = viz.label_to_color(labels)  # [N,3] in [0,1]

    # Project
    viz.save_pointcloud_projection(points, colors, cam)

if __name__ == '__main__':
    main()
