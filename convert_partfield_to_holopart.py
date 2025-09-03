"""
Convert PartField clustering output (face labels) into a multi-part GLB for HoloPart.
- Inputs:
    --labels-npy: face-wise labels (np.uint32/int) from PartField clustering
    --original-mesh: original textured mesh (e.g., .glb/.gltf/.obj)
    --output-dir: output folder
- Outputs:
    <output-dir>/parts_for_holopart.glb
    <output-dir>/original_basecolor.png  (if found)
"""

import argparse
import os
import json
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_labels_and_mesh(labels_npy: str, original_mesh: str) -> Tuple[np.ndarray, trimesh.Trimesh]:
    labels = np.load(labels_npy)
    mesh = trimesh.load(original_mesh, force="mesh")
    if mesh.faces.shape[0] != labels.shape[0]:
        raise ValueError(f"Faces ({mesh.faces.shape[0]}) != labels ({labels.shape[0]}).")
    return labels, mesh


def extract_parts_by_labels(mesh: trimesh.Trimesh, labels: np.ndarray) -> Tuple[List[trimesh.Trimesh], Dict[str, Dict]]:
    parts: List[trimesh.Trimesh] = []
    unique_labels = np.unique(labels)

    for part_id in unique_labels.tolist():
        mesh_part = mesh.submesh([labels == part_id], append=True)
        mesh_part.name = f"part_{int(part_id):02d}"
        parts.append(mesh_part)
        
    return parts


def save_original_basecolor_texture(mesh: trimesh.Trimesh, out_dir: Path) -> str | None:
    """Extract and save original textures if available."""
    texture_info = {}
    
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        
        # Check if there's a base color texture
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture = material.baseColorTexture
            
            # Save texture
            texture_path = os.path.join(out_dir, "original_basecolor.png")
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


def export_parts_scene(parts: List[trimesh.Trimesh], out_path: Path) -> str:
    scene = trimesh.Scene()
    for p in parts:
        scene.add_geometry(p, node_name=p.name, geom_name=p.name)
    scene.export(out_path)
    return str(out_path)

def build_and_save_metadata(
    labels: np.ndarray,
    mesh: trimesh.Trimesh,
    parts: List[trimesh.Trimesh],
    original_mesh_path: Path,
    labels_npy_path: Path,
    holopart_input_glb: Path,
    texture_info: Dict[str, Any],
    out_dir: Path,
) -> Path:
    """Save a compact metadata JSON for downstream tasks."""
    part_items: Dict[str, Dict[str, Any]] = {}
    for p in parts:
        # parse label from name: "part_XX"
        lab = int(p.name.split("_")[1])
        face_idx = np.where(labels == lab)[0].tolist()
        part_items[p.name] = {
            "label_id": lab,
            "face_count": len(face_idx),
            "original_face_indices": face_idx,  # minimal but useful for tracing back if needed
        }

    has_uv = bool(getattr(mesh.visual, "uv", None) is not None)

    meta = {
        "source": {
            "original_mesh": str(original_mesh_path),
            "labels_npy": str(labels_npy_path),
        },
        "holopart_input_glb": str(holopart_input_glb),
        "mesh_stats": {
            "n_vertices": int(mesh.vertices.shape[0]),
            "n_faces": int(mesh.faces.shape[0]),
            "has_uv": has_uv,
        },
        "texture": texture_info,
        "parts": part_items,
    }

    meta_path = out_dir / "conversion_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-npy", required=True, help="PartField clustering labels.npy (face-wise)")
    ap.add_argument("--original-mesh", required=True, help="Original textured mesh (.glb/.gltf/.obj)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holopart_glb = out_dir / "parts_for_holopart.glb"

    labels, mesh = load_labels_and_mesh(args.labels_npy, args.original_mesh)
    parts = extract_parts_by_labels(mesh, labels)
    export_parts_scene(parts, holopart_glb)
    texture_info = save_original_basecolor_texture(mesh, out_dir)
    
    build_and_save_metadata(
        labels=labels,
        mesh=mesh,
        parts=parts,
        original_mesh_path=Path(args.original_mesh),
        labels_npy_path=Path(args.labels_npy),
        holopart_input_glb=holopart_glb,
        texture_info=texture_info,
        out_dir=out_dir,
    )

    print(f"[OK] HoloPart input: {out_dir / 'parts_for_holopart.glb'}")
    if texture_info.get("has_texture"):
        print(f"[OK] Saved basecolor: {texture_info['texture_path']}")

if __name__ == "__main__":
    main()
