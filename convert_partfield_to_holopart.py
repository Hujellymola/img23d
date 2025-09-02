#!/usr/bin/env python3
"""
Convert PartField clustering output to HoloPart input format.
Environment: partfield conda environment

This script extracts parts from PartField labels and saves original textures.
"""

import argparse
import json
import os
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image


def load_partfield_results(labels_npy_path, original_glb_path):
    """Load PartField labels and original mesh."""
    # Load labels - assumes labels.npy from clustering output
    labels = np.load(labels_npy_path)
    
    # Load original mesh
    mesh = trimesh.load(original_glb_path, force='mesh')
    
    # Validate labels match mesh faces
    if len(labels) != len(mesh.faces):
        raise ValueError(f"Labels count ({len(labels)}) != faces count ({len(mesh.faces)})")
    
    return labels, mesh


def extract_parts_by_labels(mesh, labels):
    """Extract individual parts based on face labels."""
    unique_labels = np.unique(labels)
    parts = []
    part_metadata = {}
    
    for label_id in unique_labels:
        # Get faces for this part
        part_face_mask = (labels == label_id)
        part_face_indices = np.where(part_face_mask)[0]
        
        if len(part_face_indices) == 0:
            continue
            
        # Extract submesh for this part
        part_faces = mesh.faces[part_face_indices]
        
        # Get unique vertices used by these faces
        used_vertices = np.unique(part_faces.flatten())
        
        # Create vertex mapping from old to new indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        # Extract vertices and remap faces
        part_vertices = mesh.vertices[used_vertices]
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in part_faces])
        
        # Create part mesh
        part_mesh = trimesh.Trimesh(vertices=part_vertices, faces=remapped_faces)
        part_mesh.name = f"part_{label_id:02d}"
        
        # Store texture if available
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            # Copy visual properties
            part_mesh.visual = mesh.visual.copy()
        
        parts.append(part_mesh)
        
        # Store metadata
        part_metadata[f"part_{label_id:02d}"] = {
            "label_id": int(label_id),
            "face_count": len(part_face_indices),
            "vertex_count": len(used_vertices),
            "original_face_indices": part_face_indices.tolist()
        }
    
    return parts, part_metadata


def save_original_textures(mesh, output_dir):
    """Extract and save original textures if available."""
    texture_info = {}
    
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        
        # Check if there's a base color texture
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture = material.baseColorTexture
            
            # Save texture
            texture_path = os.path.join(output_dir, "original_texture.png")
            texture.save(texture_path)
            
            texture_info = {
                "has_texture": True,
                "texture_path": texture_path,
                "texture_size": texture.size
            }
        else:
            texture_info = {"has_texture": False}
    else:
        texture_info = {"has_texture": False}
    
    return texture_info


def create_holopart_scene(parts):
    """Create HoloPart compatible scene."""
    # HoloPart expects a Scene with named geometries
    scene = trimesh.Scene()
    
    for part in parts:
        print(f"Adding part: {part.name}")
        scene.add_geometry(part, node_name=part.name, geom_name=part.name)
    
    return scene


def main():
    parser = argparse.ArgumentParser(description="Convert PartField output to HoloPart input")
    parser.add_argument("--labels-npy", required=True, help="Path to labels.npy from PartField clustering")
    parser.add_argument("--original-glb", required=True, help="Path to original GLB file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading PartField results from {args.labels_npy}")
    labels, mesh = load_partfield_results(args.labels_npy, args.original_glb)
    
    print(f"Found {len(np.unique(labels))} unique parts")
    
    # Extract parts
    print("Extracting parts...")
    parts, part_metadata = extract_parts_by_labels(mesh, labels)
    
    # Save original textures
    print("Saving original textures...")
    texture_info = save_original_textures(mesh, args.output_dir)
    
    # Create HoloPart scene
    print("Creating HoloPart scene...")
    scene = create_holopart_scene(parts)
    
    # Save HoloPart input
    holopart_input_path = os.path.join(args.output_dir, "parts_for_holopart.glb")
    scene.export(holopart_input_path)
    
    # Save metadata
    conversion_metadata = {
        "source_files": {
            "labels_npy": args.labels_npy,
            "original_glb": args.original_glb
        },
        "part_count": len(parts),
        "parts": part_metadata,
        "texture_info": texture_info,
        "holopart_input": holopart_input_path
    }
    
    metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(conversion_metadata, f, indent=2)
    
    print(f"Conversion complete!")
    print(f"HoloPart input: {holopart_input_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Part count: {len(parts)}")


if __name__ == "__main__":
    main()
