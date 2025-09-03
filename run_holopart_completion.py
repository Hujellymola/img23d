"""
Run HoloPart completion on a multi-part GLB produced by convert_partfield_to_holopart.py.
- Inputs:
    --input-scene: parts_for_holopart.glb
    --output-dir
    (optional) --weights-dir for HoloPart
- Outputs:
    <output-dir>/completed_scene.glb
    <output-dir>/completion_metadata.json
    <output-dir>/parts/<part_xxx_completed.obj>
Notes:
- uses HoloPart's pipeline API. Ensure the HoloPart repo is in PYTHONPATH.
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
import pymeshlab
from torch_cluster import nearest

# Add HoloPart to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../dependencies/HoloPart"))
from holopart.pipelines.pipeline_holopart import HoloPartPipeline
from holopart.inference_utils import hierarchical_extract_geometry, flash_extract_geometry

NUM_SURFACE_SAMPLES = 20480
PART_NORMALIZE_SCALE = 0.7


def prepare_data(data_path, device="cuda"):
    """Prepare data for HoloPart - same as original inference_holopart.py"""
    if data_path.endswith(".glb"):
        parts_mesh = trimesh.load(data_path)
        part_name_list = []
        part_pcd_list = []
        whole_cond_list = []
        part_cond_list = []
        part_local_cond_list = []
        part_center_list = []
        part_scale_list = []
        geometry_items = sorted(parts_mesh.geometry.items(), key=lambda x: x[0])
        for i, (name, part_mesh) in enumerate(geometry_items):
            part_surface_points, face_idx = part_mesh.sample(NUM_SURFACE_SAMPLES, return_index=True)
            part_surface_normals = part_mesh.face_normals[face_idx]
            part_pcd = np.concatenate([part_surface_points, np.ones_like(part_surface_points[:, :1])*i], axis=-1)
            part_pcd_list.append(part_pcd)

            part_surface_points = torch.FloatTensor(part_surface_points)
            part_surface_normals = torch.FloatTensor(part_surface_normals)
            part_cond = torch.cat([part_surface_points, part_surface_normals], dim=-1)
            part_local_cond = part_cond.clone()
            part_cond_max = part_local_cond[:, :3].max(dim=0)[0]
            part_cond_min = part_local_cond[:, :3].min(dim=0)[0]
            part_center_new = (part_cond_max + part_cond_min) / 2
            part_local_cond[:, :3] = part_local_cond[:, :3] - part_center_new
            part_scale_new = (part_local_cond[:, :3].abs().max() / (0.95 * PART_NORMALIZE_SCALE)).item()
            part_local_cond[:, :3] = part_local_cond[:, :3] / part_scale_new
            part_cond_list.append(part_cond)
            part_local_cond_list.append(part_local_cond)
            part_name_list.append(name)
            part_center_list.append(part_center_new)
            part_scale_list.append(part_scale_new)
        
        part_pcd = np.concatenate(part_pcd_list, axis=0)
        part_pcd = torch.FloatTensor(part_pcd).to(device)
        whole_mesh = parts_mesh.dump(concatenate=True)
        whole_surface_points, face_idx = whole_mesh.sample(NUM_SURFACE_SAMPLES, return_index=True)
        whole_surface_normals = whole_mesh.face_normals[face_idx]
        whole_surface_points = torch.FloatTensor(whole_surface_points)
        whole_surface_normals = torch.FloatTensor(whole_surface_normals)
        whole_surface_points_tensor = whole_surface_points.to(device)
        nearest_idx = nearest(whole_surface_points_tensor, part_pcd[:, :3])
        nearest_part = part_pcd[nearest_idx]
        nearest_part = nearest_part[:, 3].cpu()
        
        for i in range(len(part_cond_list)):
            surface_points_part_mask = (nearest_part == i).float()
            whole_cond = torch.cat([whole_surface_points, whole_surface_normals, surface_points_part_mask[..., None]], dim=-1)
            whole_cond_list.append(whole_cond)

        batch_data = {
            "whole_cond": torch.stack(whole_cond_list, dim=0).to(device),
            "part_cond": torch.stack(part_cond_list, dim=0).to(device),
            "part_local_cond": torch.stack(part_local_cond_list, dim=0).to(device),
            "part_id_list": part_name_list,
            "part_center_list": part_center_list,
            "part_scale_list": part_scale_list,
        }
    else:
        raise ValueError("Unsupported file format. Please provide a .glb file.")
    
    return batch_data


def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    """Simplify mesh using pymeshlab - same as original"""
    mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.meshing_merge_close_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
    mesh = ms.current_mesh()
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


@torch.no_grad()
def run_holopart_with_part_saving(
    pipe: Any,
    batch: dict,
    batch_size: int,
    seed: int,
    output_dir: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.5,
    dense_octree_depth=8,
    hierarchical_octree_depth=9,
    flash_octree_depth=9,
    final_octree_depth=-1,
    num_chunks=10000,
    use_flash_decoder: bool = True,
    bounds=(-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
    post_smooth=True,
    device: str = "cuda",
):
    """Modified run_holopart that saves individual parts and original parts."""
    part_surface = batch["part_cond"]
    whole_surface = batch["whole_cond"]
    part_local_surface = batch["part_local_cond"]
    part_id_list = batch["part_id_list"]
    part_center_list = batch["part_center_list"]
    part_scale_list = batch["part_scale_list"]

    latent_list = []
    mesh_list = []
    completion_results = {}

    random_colors = np.random.rand(len(part_surface), 3)

    # Run inference in batches
    for i in range(0, len(part_surface), batch_size):
        part_surface_batch = part_surface[i : i + batch_size]
        whole_surface_batch = whole_surface[i : i + batch_size]
        part_local_surface_batch = part_local_surface[i : i + batch_size]

        meshes_latent = pipe(
            part_surface=part_surface_batch,
            whole_surface=whole_surface_batch,
            part_local_surface=part_local_surface_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            output_type="latent",
        ).samples
        latent_list.append(meshes_latent)
    
    meshes_latent = torch.cat(latent_list, dim=0)

    if use_flash_decoder:
        pipe.vae.set_flash_decoder()
    
    # Process each part
    for i, mesh_latent in enumerate(meshes_latent):
        part_name = part_id_list[i]
        mesh_latent = mesh_latent.unsqueeze(0)

        # Extract completed geometry
        if use_flash_decoder:
            output = flash_extract_geometry(
                mesh_latent,
                pipe.vae,
                bounds=bounds,
                octree_depth=flash_octree_depth,
                num_chunks=num_chunks,
            )
        else:
            geometric_func = lambda x: pipe.vae.decode(mesh_latent, sampled_points=x).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
                final_octree_depth=final_octree_depth,
                post_smooth=post_smooth
            )
        
        # Create completed mesh
        meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]
        completed_mesh = trimesh.util.concatenate(meshes)
        completed_mesh = simplify_mesh(completed_mesh, 10000)
        
        # Apply original transform
        completed_mesh.apply_scale(part_scale_list[i])
        completed_mesh.apply_translation(part_center_list[i])
        completed_mesh.name = part_name
        completed_mesh.visual.vertex_colors = random_colors[i]
        
        # Save individual completed part
        completed_path = os.path.join(output_dir, f"{part_name}_completed.obj")
        completed_mesh.export(completed_path)
        # trimesh.Scene([completed_mesh]).export(os.path.join(output_dir, f"{part_name}_completed.glb"))
        
        mesh_list.append(completed_mesh)
        
        # Store completion info
        completion_results[part_name] = {
            "completed_mesh_path": completed_path,
            "part_center": part_center_list[i].cpu().numpy().tolist(),
            "part_scale": float(part_scale_list[i]),
            "vertex_count": len(completed_mesh.vertices),
            "face_count": len(completed_mesh.faces)
        }
    
    # Create final scene
    scene = trimesh.Scene(mesh_list)
    
    return scene, completion_results


def main():
    parser = argparse.ArgumentParser(description="Run HoloPart completion with part saving")
    parser.add_argument("--input-scene", required=True, help="Input GLB scene from conversion step")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")

    args = parser.parse_args()
    
    device = "cuda"
    dtype = torch.float16

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download and load HoloPart model
    holopart_weights_dir = os.path.join(os.path.dirname(__file__), "../dependencies/HoloPart/pretrained_weights/HoloPart")
    if not os.path.exists(holopart_weights_dir):
        print("Downloading HoloPart weights...")
        snapshot_download(repo_id="VAST-AI/HoloPart", local_dir=holopart_weights_dir)

    # Initialize HoloPart pipeline
    print("Loading HoloPart pipeline...")
    pipe: HoloPartPipeline = HoloPartPipeline.from_pretrained(holopart_weights_dir).to(device, dtype)
    
    # Prepare data
    print(f"Preparing data from {args.input_scene}")
    parts_data = prepare_data(args.input_scene, device=device)
    print(f"Found {len(parts_data['part_id_list'])} parts to complete")

    # Run completion
    print("Running HoloPart completion...")
    completed_scene, completion_results = run_holopart_with_part_saving(
        pipe,
        batch=parts_data,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    # Save completed scene
    output_scene_path = os.path.join(args.output_dir, "completed_scene.glb")
    completed_scene.export(output_scene_path)

    # Save completion metadata
    completion_metadata = {
        "input_scene": args.input_scene,
        "output_scene": output_scene_path,
        "completion_settings": {
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "batch_size": args.batch_size
        },
        "completed_parts": completion_results
    }
    
    metadata_path = os.path.join(args.output_dir, "completion_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(completion_metadata, f, indent=2)

    print(f"[OK] Completed scene: {output_scene_path}")
    print(f"[OK] Per-Part OBJs saved in: {args.output_dir}")
    print(f"[OK] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
