"""
Transfer original baseColor textures to HoloPart-completed parts.
Inputs:
  --step1-dir: contains conversion_metadata.json, parts_for_holopart.glb, original_basecolor.png
  --step2-dir: contains completion_metadata.json, completed_scene.glb, parts/<part_xxx_completed.obj|glb>
  --output-dir
Key logic:
  1) Per part: build original submesh (with original UV+texture).
  2) ICP align completed part to original part (rigid).
  3) Classify completed faces by multi-sample distance vote -> old/new.
  4) Keep original faces (with original UV), keep only "new" faces from completed.
  5) Unwrap new faces with xatlas; build a horizontal 2W x H atlas:
       - left half: original texture (no resample), uv_orig'=(0.5*u, v)
       - right half: new-region uv_new'=(0.5*u+0.5, v)
  6) Export per-part textured GLB (embedded texture) + final scene GLB.
Deps: trimesh, numpy, pillow, xatlas, open3d, rtree
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw
import xatlas
import open3d as o3d
import matplotlib.pyplot as plt



# -----------------------------
# IO helpers
# -----------------------------
def _load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _load_mesh_force_mesh(p: Path) -> trimesh.Trimesh:
    m = trimesh.load(p, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"{p} is not a Trimesh")
    return m


def _submesh_by_faces(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> trimesh.Trimesh:
    return mesh.submesh([face_idx], append=True)


# -----------------------------
# ICP: align completed -> original (rigid)
# -----------------------------
def _mesh_to_o3d_pcd(mesh: trimesh.Trimesh, n_samples: int = 5000) -> o3d.geometry.PointCloud:
    pts, face_idx = mesh.sample(n_samples, return_index=True)
    norms = mesh.face_normals[face_idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
    return pcd


def _icp_align_completed_to_original(
    comp_mesh: trimesh.Trimesh,
    orig_mesh: trimesh.Trimesh,
    icp_samples: int = 5000,
    icp_ratio: float = 0.01,
    icp_max_iter: int = 60,
) -> np.ndarray:
    """Return 4x4 rigid transform to align comp_mesh -> orig_mesh."""
    diag = float(np.linalg.norm(orig_mesh.bounds[1] - orig_mesh.bounds[0]))
    thr = max(icp_ratio * diag, 1e-6)

    src = _mesh_to_o3d_pcd(comp_mesh, n_samples=icp_samples)
    tgt = _mesh_to_o3d_pcd(orig_mesh, n_samples=icp_samples)

    init = np.eye(4)
    res = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        thr,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter),
    )
    return np.asarray(res.transformation, dtype=np.float64)


# -----------------------------
# New/old face classification (vote with multi-point samples per face)
# -----------------------------
def _face_multi_sample_points(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return (F, K=4, 3): barycenter + 3 edge midpoints."""
    v = mesh.vertices
    f = mesh.faces
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    c = (v0 + v1 + v2) / 3.0
    m01 = (v0 + v1) / 2.0
    m12 = (v1 + v2) / 2.0
    m20 = (v2 + v0) / 2.0
    samples = np.stack([c, m01, m12, m20], axis=1)  # (F,4,3)
    return samples


def _compute_new_face_mask_vote(
    comp_aligned: trimesh.Trimesh,
    orig_part: trimesh.Trimesh,
    eps_ratio: float = 0.01,
    old_vote_ratio: float = 0.6,
) -> np.ndarray:
    """Return boolean new_mask for comp_aligned.faces."""
    diag = float(np.linalg.norm(orig_part.bounds[1] - orig_part.bounds[0]))
    thr = max(eps_ratio * diag, 1e-8)

    # Sample 4 points per face
    samples = _face_multi_sample_points(comp_aligned).reshape(-1, 3)
    q = trimesh.proximity.ProximityQuery(orig_part)
    d = np.abs(q.signed_distance(samples))  # (F*4,)
    d = d.reshape(-1, 4)
    close = (d < thr).sum(axis=1) / 4.0  # ratio per face in [0,1]
    # old face if majority close to original surface
    old_mask = close >= old_vote_ratio
    new_mask = ~old_mask
    return new_mask


# -----------------------------
# UV unwrap (xatlas) and atlas build (2W x H, horizontal)
# -----------------------------
def _xatlas_unwrap(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    return vmapping, indices, uvs  # uvs in [0,1]^2


def _merge_geo_and_build_atlas_horizontal(
    orig_part: trimesh.Trimesh,
    new_region: trimesh.Trimesh,
    orig_tex: Image.Image,
) -> Tuple[trimesh.Trimesh, Image.Image]:
    """Return (merged_mesh_with_uv, atlas_image). Atlas size = (2W, H)."""
    assert orig_part.visual is not None and getattr(orig_part.visual, "uv", None) is not None, \
        "Original part must have UV."

    W, H = (orig_tex.width, orig_tex.height) if orig_tex is not None else (1024, 1024)
    atlas_W, atlas_H = 2 * W, H

    # Left half: paste original texture
    atlas_img = Image.new("RGBA", (atlas_W, atlas_H), (128, 128, 128, 255))
    if orig_tex is not None:
        atlas_img.paste(orig_tex.convert("RGBA"), (0, 0))

    # Original UV -> left half (u'=0.5*u, v'=v)
    uv_orig = np.asarray(orig_part.visual.uv, dtype=np.float32).copy()
    uv_orig[:, 0] *= 0.5

    # New region unwrap -> right half: (u'=0.5*u+0.5, v'=v)
    vmapping, f_new_idx, uvs = _xatlas_unwrap(new_region)
    V1 = new_region.vertices[vmapping]
    F1 = f_new_idx.astype(np.uint32)
    uv_new = uvs.astype(np.float32).copy()
    uv_new[:, 0] = 0.5 * uv_new[:, 0] + 0.5

    # Merge geometry
    V0 = np.asarray(orig_part.vertices, dtype=np.float32)
    F0 = np.asarray(orig_part.faces, dtype=np.uint32)
    V = np.vstack([V0, V1])
    F = np.vstack([F0, F1 + np.uint32(V0.shape[0])])
    UV = np.vstack([uv_orig, uv_new])

    merged = trimesh.Trimesh(vertices=V, faces=F, process=False)
    merged.visual = trimesh.visual.texture.TextureVisuals(uv=UV, image=atlas_img)
    merged.visual.material = trimesh.visual.texture.SimpleMaterial(image=atlas_img)
    return merged, atlas_img


# -----------------------------
# Viz: UV wireframe + verts + Coverage mask
# -----------------------------
def _save_uv_debug_images(
    tex_img: Image.Image,
    uv: np.ndarray,          # (N,2) in [0,1]
    faces: np.ndarray,       # (M,3)
    out_uv_png: Path,
    out_cov_png: Path,
    title: str,
) -> None:
    tex_np = np.array(tex_img.convert("RGBA"))
    H, W = tex_np.shape[0], tex_np.shape[1]
    tris = uv[faces]  # (M,3,2)

    # Figure 1: UV wireframe + verts
    plt.figure(figsize=(8, 4.5))
    plt.imshow(tex_np, extent=[0,1,1,0])
    # edges
    for tri in tris:
        plt.plot([tri[0,0], tri[1,0]], [1-tri[0,1], 1-tri[1,1]], color='red', linewidth=0.15)
        plt.plot([tri[1,0], tri[2,0]], [1-tri[1,1], 1-tri[2,1]], color='red', linewidth=0.15)
        plt.plot([tri[2,0], tri[0,0]], [1-tri[2,1], 1-tri[0,1]], color='red', linewidth=0.15)
    # verts
    plt.scatter(uv[:,0], 1-uv[:,1], s=1, c='cyan', alpha=0.6)
    plt.title(f"{title}  (edges:red, verts:cyan)")
    plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("U"); plt.ylabel("V"); plt.grid(True, linewidth=0.2)
    out_uv_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_uv_png, dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: coverage mask
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for tri in tris:
        pts = [(tri[0,0]*W, (1-tri[0,1])*H),
               (tri[1,0]*W, (1-tri[1,1])*H),
               (tri[2,0]*W, (1-tri[2,1])*H)]
        draw.polygon(pts, fill=255)
    mask_np = np.array(mask)

    plt.figure(figsize=(8, 4.5))
    plt.imshow(tex_np, extent=[0,1,1,0])
    plt.imshow(mask_np, cmap='Greens', alpha=0.35, extent=[0,1,1,0], vmin=0, vmax=255)
    plt.title(f"{title}  (coverage: green)")
    plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("U"); plt.ylabel("V")
    plt.savefig(out_cov_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    
# -----------------------------
# Export helpers
# -----------------------------
def _export_part_textured(mesh: trimesh.Trimesh, atlas: Image.Image, out_glb: Path, out_png: Path, node_name: str) -> None:
    mesh = mesh.copy()
    mesh.name = node_name
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_png)
    trimesh.Scene([mesh]).export(str(out_glb))


def _assemble_scene(parts_glb: List[Path], out_scene: Path) -> None:
    scene = trimesh.Scene()
    for p in parts_glb:
        m = _load_mesh_force_mesh(p)  # ensure as single geometry
        scene.add_geometry(m, node_name=p.stem)  # unique by filename
    out_scene.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_scene))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Refined texture transfer for HoloPart-completed parts")
    ap.add_argument("--step1-dir", default="output/step1_conversion")
    ap.add_argument("--step2-dir", default="output/step2_completion")
    ap.add_argument("--output-dir", default="output/step3_tex_transfer")
    # thresholds
    ap.add_argument("--eps-ratio", type=float, default=0.01, help="distance threshold as fraction of bbox diag")
    ap.add_argument("--old-vote-ratio", type=float, default=0.6, help="majority ratio to keep a face as 'old'")
    # ICP params
    ap.add_argument("--icp-samples", type=int, default=5000)
    ap.add_argument("--icp-ratio", type=float, default=0.01, help="icp correspondence distance = icp_ratio * diag")
    ap.add_argument("--icp-max-iter", type=int, default=60)
    args = ap.parse_args()

    step1 = Path(args.step1_dir)
    step2 = Path(args.step2_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    conv_meta = _load_json(step1 / "conversion_metadata.json")
    comp_meta = _load_json(step2 / "completion_metadata.json")

    original_mesh_path = Path(conv_meta["source"]["original_mesh"])
    original_mesh = _load_mesh_force_mesh(original_mesh_path)

    tex_path = Path(conv_meta.get("texture", {}).get("texture_path", "")) if "texture" in conv_meta else Path()
    orig_tex = Image.open(tex_path).convert("RGBA") if tex_path.exists() else Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))

    # map part -> face indices from step1
    part_to_faces: Dict[str, List[int]] = {
        pname: pdata.get("original_face_indices", [])
        for pname, pdata in conv_meta.get("parts", {}).items()
    }

    comp_parts = comp_meta.get("completed_parts", {})
    sorted_part_names = sorted(comp_parts.keys())

    parts_out_dir = outdir / "parts_textured"
    parts_out_dir.mkdir(parents=True, exist_ok=True)

    part_output_glbs: List[Path] = []
    report: Dict[str, dict] = {}

    for pname in sorted_part_names:
        cinfo = comp_parts[pname]
        comp_path = Path(cinfo["completed_mesh_path"])
        if not comp_path.exists():
            continue

        face_idx = np.array(part_to_faces.get(pname, []), dtype=np.int64)
        if face_idx.size == 0:
            continue

        # 1) original submesh (keeps original UV)
        orig_part = _submesh_by_faces(original_mesh, face_idx)

        # 2) load completed and ICP align to original
        comp_part = _load_mesh_force_mesh(comp_path)
        T = _icp_align_completed_to_original(
            comp_part, orig_part,
            icp_samples=args.icp_samples,
            icp_ratio=args.icp_ratio,
            icp_max_iter=args.icp_max_iter
        )
        comp_part_aligned = comp_part.copy()
        comp_part_aligned.apply_transform(T)

        # 3) classify faces by vote
        new_mask = _compute_new_face_mask_vote(
            comp_part_aligned, orig_part,
            eps_ratio=args.eps_ratio,
            old_vote_ratio=args.old_vote_ratio
        )
        part_glb = parts_out_dir / f"{pname}_textured.glb"
        part_png = parts_out_dir / f"{pname}_texture.png"
        part_uv_png = parts_out_dir / f"{pname}_uv_overlay.png"
        part_cov_png = parts_out_dir / f"{pname}_coverage_overlay.png"
        
        if not np.any(new_mask):
            # No new faces: keep original part as-is with original texture
            # Save debug texture (original)
            orig_tex.save(part_png)
            _save_uv_debug_images(orig_tex, np.asarray(orig_part.visual.uv, np.float32),
                                  np.asarray(orig_part.faces, np.int64),
                                  part_uv_png, part_cov_png, title=f"{pname} (original only)")
            trimesh.Scene([orig_part.copy()]).export(str(part_glb))
            part_output_glbs.append(part_glb)
            report[pname] = {
                "mode": "no_new_faces",
                "original_faces": int(len(orig_part.faces)),
                "new_faces": 0,
                "texture": str(part_png)
            }
            continue

        # 4) extract new region
        new_faces = np.where(new_mask)[0]
        new_region = comp_part_aligned.submesh([new_faces], append=True)

        # 5) build merged mesh + horizontal atlas
        merged, atlas_img = _merge_geo_and_build_atlas_horizontal(orig_part, new_region, orig_tex)

        # 6) export
        part_glb = parts_out_dir / f"{pname}_textured.glb"
        part_png = parts_out_dir / f"{pname}_texture.png"
        _export_part_textured(merged, atlas_img, part_glb, part_png, node_name=pname)
        _save_uv_debug_images(atlas_img,
                              np.asarray(merged.visual.uv, np.float32),
                              np.asarray(merged.faces, np.int64),
                              part_uv_png, part_cov_png, title=f"{pname} (orig-left, new-right)")
        part_output_glbs.append(part_glb)
        report[pname] = {
            "mode": "merged_original_plus_new",
            "original_faces": int(len(orig_part.faces)),
            "new_faces": int(len(new_region.faces)),
            "texture": str(part_png)
        }

    # Assemble final scene (unique node names by filename stem)
    final_scene = outdir / "completed_textured_scene.glb"
    _assemble_scene(part_output_glbs, final_scene)

    # Write metadata
    meta = {
        "inputs": {
            "step1_dir": str(step1),
            "step2_dir": str(step2),
            "original_mesh": str(original_mesh_path),
            "original_texture": str(tex_path) if tex_path.exists() else None
        },
        "outputs": {
            "scene_glb": str(final_scene),
            "parts_dir": str(parts_out_dir)
        },
        "params": {
            "eps_ratio": args.eps_ratio,
            "old_vote_ratio": args.old_vote_ratio,
            "icp_samples": args.icp_samples,
            "icp_ratio": args.icp_ratio,
            "icp_max_iter": args.icp_max_iter
        },
        "parts": report
    }
    with open(outdir / "texture_transfer_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Scene: {final_scene}")
    print(f"[OK] Parts textured dir: {parts_out_dir}")
    print(f"[OK] Metadata: {outdir / 'texture_transfer_metadata.json'}")


if __name__ == "__main__":
    main()
