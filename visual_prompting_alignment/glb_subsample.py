import trimesh
import numpy as np
import open3d as o3d
import argparse
import os
from pathlib import Path

def load_mesh_from_glb(glb_path):
    """Load mesh from GLB file using trimesh."""
    try:
        mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, combine all geometries
            mesh = mesh.dump().sum()
        return mesh
    except Exception as e:
        print(f"Error loading GLB file: {e}")
        return None

def subsample_mesh_faces(mesh, max_faces=50000):
    """Subsample mesh faces to reduce complexity."""
    if len(mesh.faces) <= max_faces:
        return mesh
    
    # Calculate face areas for weighted sampling
    face_areas = mesh.area_faces
    probabilities = face_areas / face_areas.sum()
    
    # Sample faces based on area weights
    selected_faces = np.random.choice(
        len(mesh.faces), 
        size=max_faces, 
        replace=False, 
        p=probabilities
    )
    
    # Get unique vertices from selected faces
    selected_face_indices = mesh.faces[selected_faces]
    unique_vertices = np.unique(selected_face_indices.flatten())
    
    # Create vertex mapping
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
    
    # Create new mesh with subsampled faces
    new_vertices = mesh.vertices[unique_vertices]
    new_faces = np.array([[vertex_map[v] for v in face] for face in selected_face_indices])
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

def mesh_to_point_cloud(mesh, n_points=10000):
    """Convert mesh to point cloud by sampling points on surface."""
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    # Get normals at sampled points
    if mesh.face_normals is not None:
        normals = mesh.face_normals[face_indices]
    else:
        normals = np.zeros_like(points)
    
    return points, normals

def save_point_cloud(points, normals, output_path):
    """Save point cloud to PLY format."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Estimate normals if they're zero
    if np.allclose(normals, 0):
        pcd.estimate_normals()
    
    o3d.io.write_point_cloud(str(output_path), pcd)

def save_simplified_mesh(mesh, output_path):
    """Save simplified mesh to OBJ format."""
    mesh.export(str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Subsample GLB files for PartField processing")
    parser.add_argument("input_path", help="Input GLB file or directory containing GLB files")
    parser.add_argument("output_dir", help="Output directory for processed files")
    parser.add_argument("--max_faces", type=int, default=50000, help="Maximum number of faces for mesh subsampling")
    parser.add_argument("--n_points", type=int, default=20000, help="Number of points for point cloud conversion")
    parser.add_argument("--output_format", choices=["mesh", "pointcloud", "both"], default="pointcloud", 
                       help="Output format: mesh (OBJ), pointcloud (PLY), or both")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible subsampling")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single file or directory
    input_path = Path(args.input_path)
    if input_path.is_file():
        glb_files = [input_path]
    else:
        glb_files = list(input_path.glob("*.glb"))
    
    if not glb_files:
        print("No GLB files found!")
        return
    
    for glb_file in glb_files:
        print(f"Processing {glb_file.name}...")
        
        # Load mesh
        mesh = load_mesh_from_glb(glb_file)
        if mesh is None:
            print(f"Failed to load {glb_file.name}, skipping...")
            continue
        
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Subsample mesh if it's too large
        if len(mesh.faces) > args.max_faces:
            mesh = subsample_mesh_faces(mesh, args.max_faces)
            print(f"Subsampled mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Save outputs based on format choice
        base_name = glb_file.stem
        
        if args.output_format in ["mesh", "both"]:
            mesh_output_path = output_dir / f"{base_name}_simplified.obj"
            save_simplified_mesh(mesh, mesh_output_path)
            print(f"Saved simplified mesh to {mesh_output_path}")
        
        if args.output_format in ["pointcloud", "both"]:
            points, normals = mesh_to_point_cloud(mesh, args.n_points)
            pc_output_path = output_dir / f"{base_name}.ply"
            save_point_cloud(points, normals, pc_output_path)
            print(f"Saved point cloud ({len(points)} points) to {pc_output_path}")
        
        print(f"Finished processing {glb_file.name}\n")

if __name__ == "__main__":
    main()