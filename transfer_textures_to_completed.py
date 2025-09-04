"""
Single-GLB, watertight texture transfer.

Per part:
  1) (Optional) constrained alignment of completed -> original part (default: none).
  2) Face-wise coverage via centroid (+ optional edge midpoints) & normal agreement.
  3) Covered faces: per-face-consistent barycentric UV from ONE source triangle
     on the original mesh, sampling original baseColor texture (preserves look).
  4) New faces: unwrap only those faces with xatlas to a separate gray texture
     (ready for InTeX inpainting).

Outputs:
  <out>/completed_textured_scene.glb            # single GLB containing all parts
  <out>/parts_textured/<part>_new_texture.png   # gray, for InTeX
  <out>/parts_textured/*_uv_overlay.png         # UV edges/verts overlay
  <out>/parts_textured/*_coverage.png           # UV coverage overlay
  <out>/texture_transfer_metadata.json

Notes:
  - Geometry always comes from the completed mesh (watertight preserved).
  - Each part becomes two primitives/materials internally: covered (original texture), new (gray texture).
  - You can later inpaint only the new textures with the exported coverage as mask.
"""

from __future__ import annotations
import argparse
import json
import trimesh
import xatlas
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw


# =========================
# I/O utilities
# =========================

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_mesh_force_mesh(path: Path) -> trimesh.Trimesh:
    """Load as Trimesh or fail (we always want a mesh, not a Scene)."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{path} is not a Trimesh")
    return mesh


def submesh_by_faces(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> trimesh.Trimesh:
    """Extract a face-indexed submesh (shared-vertex, topology-preserving)."""
    return mesh.submesh([face_idx], append=True)


# =========================
# Alignment (optional, constrained)
# =========================

def pca_symmetry_hint(mesh: trimesh.Trimesh) -> str:
    """
    Very rough symmetry hint using PCA eigenvalues:
      - 'sphere': λ1≈λ2≈λ3
      - 'cylinder': λ1≈λ2 >> λ3
      - 'plate': λ1 >> λ2≈λ3
      - else 'generic'
    """
    V = mesh.vertices - mesh.vertices.mean(0)
    cov = np.cov(V.T)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(w)  # w0 <= w1 <= w2
    eps = 1e-12
    r01 = (w[1] + eps) / (w[0] + eps)
    r12 = (w[2] + eps) / (w[1] + eps)
    if abs(r01 - 1.0) < 0.2 and abs(r12 - 1.0) < 0.2:
        return "sphere"
    if abs(r01 - 1.0) < 0.2 and r12 > 2.5:
        return "cylinder"
    if r01 > 2.5 and abs(r12 - 1.0) < 0.2:
        return "plate"
    return "generic"


def rms_chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 4000) -> float:
    """Estimate RMS bidirectional chamfer distance by sampling."""
    pa = a.sample(n)
    pb = b.sample(n)
    closest_a, _, _ = b.nearest.on_surface(pa)
    closest_b, _, _ = a.nearest.on_surface(pb)
    da = np.linalg.norm(pa - closest_a, axis=1).mean()
    db = np.linalg.norm(pb - closest_b, axis=1).mean()
    return 0.5 * (da + db)


def auto_align_transform(
    comp: trimesh.Trimesh,
    ref: trimesh.Trimesh,
    diag: float,
    mode: str = "none",
    icp_ratio: float = 0.01,
    icp_iter: int = 20,
) -> np.ndarray:
    """
    Compute a gentle alignment transform (4x4) from completed -> original.

    mode:
      - "none": identity (recommended default; HoloPart already returns world coords).
      - "auto": only align if RMS > 0.01 * diag; if symmetric shape, translation-only; else small-step ICP.
      - "icp" : always do small-step ICP.

    Small-step ICP uses point-to-plane, tight correspondence distance (icp_ratio*diag),
    and limited iterations to avoid over-rotation on near-symmetric shapes.
    """
    if mode == "none":
        return np.eye(4)

    if mode == "auto":
        if rms_chamfer(comp, ref) <= 0.01 * diag:
            return np.eye(4)
        hint = pca_symmetry_hint(ref)
        if hint in ("sphere", "cylinder", "plate"):
            # Translation-only using mean nearest offset
            pa = comp.sample(6000)
            y, _, _ = ref.nearest.on_surface(pa)
            t = (y - pa).mean(axis=0)
            T = np.eye(4)
            T[:3, 3] = t
            return T
        # else fall through to small-step ICP

    # Small-step point-to-plane ICP
    thr = max(icp_ratio * diag, 1e-6)

    def to_pcd(m: trimesh.Trimesh, n: int = 6000) -> o3d.geometry.PointCloud:
        pts, fidx = m.sample(n, return_index=True)
        norms = m.face_normals[fidx]
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        p.normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
        return p

    src, tgt = to_pcd(comp), to_pcd(ref)
    res = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        thr,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iter),
    )
    return np.asarray(res.transformation, dtype=np.float64)


# =========================
# Coverage (face-level, consistent per face)
# =========================

def closest_on_surface(mesh: trimesh.Trimesh, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-point:
      - closest point on mesh (m,3)
      - triangle index (m,)
      - distance (m,)
    """
    closest, dist, tri_idx = mesh.nearest.on_surface(points)  # trimesh returns 3 items
    return closest, tri_idx, dist


def face_vote_covered(
    comp: trimesh.Trimesh,
    ref: trimesh.Trimesh,
    eps_ratio: float,
    cos_thr: float,
    vote: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decide if each completed face is 'covered' (belongs to original visible region) vs 'new'.

    Strategy:
      - Sample centroid (+ optionally three edge midpoints) of each face.
      - For samples: require distance-to-ref < eps AND normal alignment cos > cos_thr.
      - Majority vote across samples for a face -> covered/uncovered.
      - Record the reference triangle index of the centroid sample (used for UV transfer).

    Returns:
      covered_mask: (F,) boolean
      covered_src_tri: (F,) int (only meaningful where covered_mask==True)
    """
    faces = comp.faces
    verts = comp.vertices
    fnorm = comp.face_normals

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    samples = [centroids]
    if vote:
        samples += [(v0 + v1) / 2.0, (v1 + v2) / 2.0, (v2 + v0) / 2.0]

    S = np.stack(samples, axis=1)  # (F,K,3)
    S_flat = S.reshape(-1, 3)

    closest, tri_idx, dist = closest_on_surface(ref, S_flat)
    ref_fn = ref.face_normals[tri_idx]

    fn = np.repeat(fnorm, repeats=S.shape[1], axis=0)
    cosang = (fn * ref_fn).sum(axis=1) / (np.linalg.norm(fn, axis=1) * np.linalg.norm(ref_fn, axis=1) + 1e-12)

    diag = float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
    dthr = max(eps_ratio * diag, 1e-8)

    good = (dist < dthr) & (cosang > cos_thr)
    good = good.reshape(-1, S.shape[1])  # (F,K)
    votes = good.sum(axis=1)

    covered_mask = votes >= (S.shape[1] // 2 + 1)  # majority
    covered_src_tri = tri_idx.reshape(-1, S.shape[1])[:, 0]  # centroid source tri
    return covered_mask, covered_src_tri


def barycentric_from_tri(tri_xyz: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute barycentric coords (clamped & renormalized) of point p in triangle tri_xyz."""
    w = trimesh.triangles.points_to_barycentric(tri_xyz[None, ...], p[None, ...])[0]
    w = np.clip(w, 0.0, 1.0)
    s = w.sum()
    return w / (s + 1e-12)


def uv_from_src_face(ref: trimesh.Trimesh, tri_id: int, verts3: np.ndarray) -> np.ndarray:
    """
    For one completed face (three 3D verts):
      - Use ONE source triangle on the original mesh (tri_id) for the entire face.
      - For each corner vertex 3D position, compute barycentric on that same source triangle.
      - Interpolate UV from the source triangle UVs.

    This ensures per-face UV consistency and eliminates striped artifacts.
    """
    tri_xyz = ref.triangles[tri_id]                # (3,3)
    uv_tri = ref.visual.uv[ref.faces[tri_id]]      # (3,2)
    uvs = []
    for p in verts3:
        # Use closest point ON that same triangle to stabilize barycentric
        cp = trimesh.triangles.closest_point(tri_xyz[None, ...], p[None, ...])[0]
        w = barycentric_from_tri(tri_xyz, cp)
        uvs.append((uv_tri * w[:, None]).sum(axis=0))
    return np.asarray(uvs, dtype=np.float32)       # (3,2)


# =========================
# Unwrap only "new" faces
# =========================

def unwrap_subset(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run xatlas on a submesh (only new faces).
    Returns: (V_unwrapped, F_unwrapped, UV_unwrapped)
    """
    sub = mesh.submesh([face_idx], append=True)
    V = np.asarray(sub.vertices, dtype=np.float32)
    F = np.asarray(sub.faces, dtype=np.uint32)
    vm, idx, uvs = xatlas.parametrize(V, F)
    return V[vm], idx.astype(np.uint32), uvs.astype(np.float32)


# =========================
# Visualization (overlays)
# =========================

def save_uv_overlays(
    tex_img: Image.Image,
    uv: np.ndarray,
    faces: np.ndarray,
    out_edges_png: Path,
    out_cov_png: Path,
    title: str,
) -> None:
    """Save UV edges/verts overlay and filled coverage visualization."""
    tex = np.array(tex_img.convert("RGBA"))
    tris = uv[faces]

    # Edges & vertices overlay
    plt.figure(figsize=(8, 6))
    plt.imshow(tex, extent=[0, 1, 1, 0])
    for tri in tris:
        plt.plot([tri[0, 0], tri[1, 0]], [1 - tri[0, 1], 1 - tri[1, 1]], "r-", lw=0.2)
        plt.plot([tri[1, 0], tri[2, 0]], [1 - tri[1, 1], 1 - tri[2, 1]], "r-", lw=0.2)
        plt.plot([tri[2, 0], tri[0, 0]], [1 - tri[2, 1], 1 - tri[0, 1]], "r-", lw=0.2)
    plt.scatter(uv[:, 0], 1 - uv[:, 1], s=0.5, c="cyan", alpha=0.6)
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    out_edges_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_edges_png, dpi=300, bbox_inches="tight")
    plt.close()

    # Coverage overlay
    H, W = tex.shape[0], tex.shape[1]
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for tri in tris:
        pts = [
            (tri[0, 0] * W, (1 - tri[0, 1]) * H),
            (tri[1, 0] * W, (1 - tri[1, 1]) * H),
            (tri[2, 0] * W, (1 - tri[2, 1]) * H),
        ]
        draw.polygon(pts, fill=255)
    m = np.array(mask)

    plt.figure(figsize=(8, 6))
    plt.imshow(tex, extent=[0, 1, 1, 0])
    plt.imshow(m, cmap="Greens", alpha=0.35, extent=[0, 1, 1, 0], vmin=0, vmax=255)
    plt.title(f"{title} coverage")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(out_cov_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-GLB watertight texture transfer (per-face-consistent UV)."
    )
    parser.add_argument("--step1-dir", required=False, default="output/step1_conversion")
    parser.add_argument("--step2-dir", required=False, default="output/step2_completion")
    parser.add_argument("--output-dir", required=False, default="output/step3_tex_transfer")

    # Coverage / normals thresholds
    parser.add_argument("--eps-ratio", type=float, default=0.015, help="distance threshold as fraction of bbox diag")
    parser.add_argument("--cos-threshold", type=float, default=0.5, help="cosine(face, ref_face) threshold")
    parser.add_argument("--vote", action="store_true", help="use centroid + edge midpoints majority vote")

    # Alignment mode (recommended default: none)
    parser.add_argument("--align-mode", choices=("none", "auto", "icp"), default="none")
    parser.add_argument("--icp-ratio", type=float, default=0.01)
    parser.add_argument("--icp-iter", type=int, default=20)

    args = parser.parse_args()

    step1 = Path(args.step1_dir)
    step2 = Path(args.step2_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    parts_dir = outdir / "parts_textured"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata from step1 (conversion) & step2 (completion)
    conv = load_json(step1 / "conversion_metadata.json")
    comp = load_json(step2 / "completion_metadata.json")

    # Original mesh + original baseColor texture
    orig_mesh_path = Path(conv["source"]["original_mesh"])
    orig_mesh = load_mesh_force_mesh(orig_mesh_path)
    tex_path = Path(conv.get("texture", {}).get("texture_path", "")) if "texture" in conv else Path()
    orig_tex = Image.open(tex_path).convert("RGBA") if tex_path.exists() \
        else Image.new("RGBA", (1024, 1024), (128, 128, 128, 255))

    # Part -> original faces (from step1 conversion metadata)
    part_to_faces: Dict[str, List[int]] = {
        k: v.get("original_face_indices", []) for k, v in conv.get("parts", {}).items()
    }

    # Build final scene with two primitives per part (covered/new)
    final_scene = trimesh.Scene()
    meta_parts: Dict[str, dict] = {}

    for part_name, info in sorted(comp.get("completed_parts", {}).items()):
        comp_path = Path(info["completed_mesh_path"])
        if not comp_path.exists():
            continue

        face_idx = np.array(part_to_faces.get(part_name, []), dtype=np.int64)
        if face_idx.size == 0:
            continue

        # Original part submesh (to read source UVs/triangles)
        ref_part = submesh_by_faces(orig_mesh, face_idx)
        if getattr(ref_part.visual, "uv", None) is None:
            print(f"[WARN] {part_name}: original part has no UV; skipped.")
            continue

        # Completed part (already in world coords via your HoloPart inverse-normalization)
        comp_part = load_mesh_force_mesh(comp_path)

        # Optional gentle alignment (default: none)
        diag = float(np.linalg.norm(ref_part.bounds[1] - ref_part.bounds[0]))
        T = auto_align_transform(
            comp_part,
            ref_part,
            diag,
            mode=args.align_mode,
            icp_ratio=args.icp_ratio,
            icp_iter=args.icp_iter,
        )
        comp_aligned = comp_part.copy()
        comp_aligned.apply_transform(T)

        # Face-wise coverage mask + source triangle per face
        covered_mask, covered_src_tri = face_vote_covered(
            comp_aligned, ref_part, args.eps_ratio, args.cos_threshold, vote=args.vote
        )

        faces = np.asarray(comp_aligned.faces, dtype=np.int64)
        verts = np.asarray(comp_aligned.vertices, dtype=np.float32)

        # Covered primitive (original texture)
        covered_V, covered_F, covered_UV = [], [], []
        vbase = 0
        for fi in np.where(covered_mask)[0]:
            vids = faces[fi]
            pts3 = verts[vids]
            uvs = uv_from_src_face(ref_part, int(covered_src_tri[fi]), pts3)
            for k in range(3):
                covered_V.append(pts3[k])
                covered_UV.append(uvs[k])
            covered_F.append([vbase, vbase + 1, vbase + 2])
            vbase += 3

        if covered_F:
            covered_mesh = trimesh.Trimesh(
                vertices=np.asarray(covered_V, np.float32),
                faces=np.asarray(covered_F, np.uint32),
                process=False,
            )
            covered_mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.asarray(covered_UV, np.float32), image=orig_tex
            )
            covered_mesh.visual.material = trimesh.visual.texture.SimpleMaterial(image=orig_tex)
            final_scene.add_geometry(covered_mesh, node_name=f"{part_name}__covered")

            # Overlays for debugging
            save_uv_overlays(
                orig_tex,
                np.asarray(covered_UV, np.float32),
                np.asarray(covered_F, np.int64),
                parts_dir / f"{part_name}_covered_uv_overlay.png",
                parts_dir / f"{part_name}_covered_coverage.png",
                f"{part_name} covered (original texture)",
            )

        # New primitive (xatlas unwrap) + gray texture for InTeX
        new_ids = np.where(~covered_mask)[0]
        new_tex_path: Path | None = None
        if new_ids.size > 0:
            V1, F1, UV1 = unwrap_subset(comp_aligned, new_ids)
            new_img = Image.new("RGBA", (orig_tex.width, orig_tex.height), (128, 128, 128, 255))
            new_tex_path = parts_dir / f"{part_name}_new_texture.png"
            new_img.save(new_tex_path)

            new_mesh = trimesh.Trimesh(vertices=V1, faces=F1, process=False)
            new_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=UV1, image=new_img)
            new_mesh.visual.material = trimesh.visual.texture.SimpleMaterial(image=new_img)
            final_scene.add_geometry(new_mesh, node_name=f"{part_name}__new")

            save_uv_overlays(
                new_img,
                np.asarray(UV1, np.float32),
                np.asarray(F1, np.int64),
                parts_dir / f"{part_name}_new_uv_overlay.png",
                parts_dir / f"{part_name}_new_coverage.png",
                f"{part_name} new (xatlas)",
            )

        meta_parts[part_name] = {
            "faces_total": int(len(comp_aligned.faces)),
            "faces_covered": int(covered_mask.sum()),
            "faces_new": int((~covered_mask).sum()),
            "new_texture": str(new_tex_path) if new_ids.size > 0 else None,
        }

    # Export single GLB scene
    final_glb = outdir / "completed_textured_scene.glb"
    final_scene.export(str(final_glb))

    # Metadata
    meta = {
        "inputs": {
            "original_mesh": str(orig_mesh_path),
            "original_texture": str(tex_path) if tex_path.exists() else None,
            "step1_dir": str(step1),
            "step2_dir": str(step2),
        },
        "outputs": {"scene_glb": str(final_glb), "parts_dir": str(parts_dir)},
        "params": {
            "eps_ratio": args.eps_ratio,
            "cos_threshold": args.cos_threshold,
            "vote": bool(args.vote),
            "align_mode": args.align_mode,
            "icp_ratio": args.icp_ratio,
            "icp_iter": args.icp_iter,
        },
        "parts": meta_parts,
    }
    with open(outdir / "texture_transfer_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] single GLB: {final_glb}")
    print(f"[OK] new textures & overlays in: {parts_dir}")


if __name__ == "__main__":
    main()
