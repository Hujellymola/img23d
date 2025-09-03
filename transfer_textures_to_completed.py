"""
Single-GLB, watertight texture transfer:
- For each part:
  1) ICP align completed -> original part
  2) Face-wise coverage (centroid + edge-midpoint vote, normal check)
  3) Covered faces: per-face-consistent barycentric UV from ONE source triangle (use original texture)
  4) New faces: xatlas unwrap to its own new texture (gray; for InTeX)
- Outputs:
  <out>/completed_textured_scene.glb          # only one GLB
  <out>/parts_textured/<part>_new_texture.png # for InTeX
  <out>/parts_textured/*_uv_overlay.png & *_coverage.png
  <out>/texture_transfer_metadata.json
Deps: trimesh, numpy, pillow, xatlas, open3d, rtree, matplotlib
"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import trimesh
from PIL import Image, ImageDraw
import xatlas, open3d as o3d
import matplotlib.pyplot as plt

# ---------------- IO ----------------
def _load_json(p: Path) -> dict:
    with open(p, "r") as f: return json.load(f)

def _load_mesh_force_mesh(p: Path) -> trimesh.Trimesh:
    m = trimesh.load(p, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"{p} is not a Trimesh")
    return m

def _submesh_by_faces(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> trimesh.Trimesh:
    return mesh.submesh([face_idx], append=True)

# ------------- ICP ------------------
def _pcd_from_mesh(mesh: trimesh.Trimesh, n_samples=6000):
    pts, fidx = mesh.sample(n_samples, return_index=True)
    norms = mesh.face_normals[fidx]
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
    return pcd

def _icp_align(comp: trimesh.Trimesh, ref: trimesh.Trimesh, icp_samples, icp_ratio, icp_max_iter) -> np.ndarray:
    diag = float(np.linalg.norm(ref.bounds[1]-ref.bounds[0]))
    thr = max(icp_ratio*diag, 1e-6)
    src = _pcd_from_mesh(comp, icp_samples)
    tgt = _pcd_from_mesh(ref,  icp_samples)
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, thr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter)
    )
    return np.asarray(res.transformation, np.float64)

# ------ coverage (face-level) -------
def _closest_on_surface(mesh: trimesh.Trimesh, points: np.ndarray):
    closest, dist, tri_idx = mesh.nearest.on_surface(points)  # 3 returns
    return closest, tri_idx, dist

def _face_vote_covered(comp: trimesh.Trimesh, ref: trimesh.Trimesh, eps_ratio: float, cos_thr: float, vote: bool=True):
    faces, verts, fnorm = comp.faces, comp.vertices, comp.face_normals
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    cen = (v0+v1+v2)/3.0
    samples = [cen]
    if vote:
        samples += [(v0+v1)/2.0, (v1+v2)/2.0, (v2+v0)/2.0]
    S = np.stack(samples, axis=1)  # (F,K,3)
    Sf = S.reshape(-1,3)
    closest, tri_idx, dist = _closest_on_surface(ref, Sf)
    ref_fn = ref.face_normals[tri_idx]
    fn = np.repeat(fnorm, S.shape[1], axis=0)
    cosang = (fn*ref_fn).sum(axis=1)/(np.linalg.norm(fn,axis=1)*np.linalg.norm(ref_fn,axis=1)+1e-12)

    diag = float(np.linalg.norm(ref.bounds[1]-ref.bounds[0]))
    dthr = max(eps_ratio*diag, 1e-8)
    close = (dist<dthr) & (cosang>cos_thr)
    close = close.reshape(-1, S.shape[1])
    votes = close.sum(axis=1)
    covered_mask = votes >= (S.shape[1]//2 + 1)
    covered_src_tri = tri_idx.reshape(-1, S.shape[1])[:,0]
    return covered_mask, covered_src_tri

def _barycentric_from_tri(tri_xyz: np.ndarray, p: np.ndarray) -> np.ndarray:
    w = trimesh.triangles.points_to_barycentric(tri_xyz[None,...], p[None,...])[0]
    w = np.clip(w, 0.0, 1.0); s = w.sum(); return w/(s+1e-12)

def _uv_from_src_face(ref: trimesh.Trimesh, tri_id: int, pts3: np.ndarray) -> np.ndarray:
    tri_xyz = ref.triangles[tri_id]                 # (3,3)
    uv_tri = ref.visual.uv[ref.faces[tri_id]]       # (3,2)
    uvs=[]
    for p in pts3:
        w = _barycentric_from_tri(tri_xyz, p)
        uvs.append((uv_tri*w[:,None]).sum(axis=0))
    return np.asarray(uvs, np.float32)

# --------- unwrap subset ----------
def _unwrap_subset(mesh: trimesh.Trimesh, face_idx: np.ndarray):
    sub = mesh.submesh([face_idx], append=True)
    V = np.asarray(sub.vertices, np.float32)
    F = np.asarray(sub.faces, np.uint32)
    vm, idx, uvs = xatlas.parametrize(V, F)
    return V[vm], idx.astype(np.uint32), uvs.astype(np.float32)

# --------- UV overlays -------------
def _save_uv_overlays(tex_img: Image.Image, uv: np.ndarray, faces: np.ndarray,
                      out_edges_png: Path, out_cov_png: Path, title: str):
    tex = np.array(tex_img.convert("RGBA")); H,W = tex.shape[0], tex.shape[1]
    tris = uv[faces]
    # edges/verts
    plt.figure(figsize=(8,6)); plt.imshow(tex, extent=[0,1,1,0])
    for tri in tris:
        plt.plot([tri[0,0],tri[1,0]],[1-tri[0,1],1-tri[1,1]],'r-',lw=0.2)
        plt.plot([tri[1,0],tri[2,0]],[1-tri[1,1],1-tri[2,1]],'r-',lw=0.2)
        plt.plot([tri[2,0],tri[0,0]],[1-tri[2,1],1-tri[0,1]],'r-',lw=0.2)
    plt.scatter(uv[:,0],1-uv[:,1],s=0.5,c='cyan',alpha=0.6)
    plt.title(title); plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal')
    out_edges_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_edges_png,dpi=300,bbox_inches='tight'); plt.close()
    # coverage
    mask = Image.new("L",(W,H),0); draw = ImageDraw.Draw(mask)
    for tri in tris:
        pts=[(tri[0,0]*W,(1-tri[0,1])*H),(tri[1,0]*W,(1-tri[1,1])*H),(tri[2,0]*W,(1-tri[2,1])*H)]
        draw.polygon(pts, fill=255)
    m=np.array(mask)
    plt.figure(figsize=(8,6)); plt.imshow(tex, extent=[0,1,1,0])
    plt.imshow(m,cmap='Greens',alpha=0.35,extent=[0,1,1,0],vmin=0,vmax=255)
    plt.title(title+" coverage"); plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal')
    plt.savefig(out_cov_png,dpi=300,bbox_inches='tight'); plt.close()

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Single-GLB watertight texture transfer (per-face-consistent UV).")
    ap.add_argument("--step1-dir", required=True)
    ap.add_argument("--step2-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eps-ratio", type=float, default=0.01)      # 0.01~0.02 更稳
    ap.add_argument("--cos-threshold", type=float, default=0.5)   # 0.3~0.6 视对齐度调
    ap.add_argument("--vote", action="store_true", help="use centroid+edges majority vote")
    ap.add_argument("--icp-samples", type=int, default=6000)
    ap.add_argument("--icp-ratio", type=float, default=0.01)
    ap.add_argument("--icp-max-iter", type=int, default=60)
    args = ap.parse_args()

    step1, step2, outdir = Path(args.step1_dir), Path(args.step2_dir), Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    parts_out = outdir / "parts_textured"; parts_out.mkdir(parents=True, exist_ok=True)

    conv = _load_json(step1 / "conversion_metadata.json")
    comp = _load_json(step2 / "completion_metadata.json")

    orig_mesh_path = Path(conv["source"]["original_mesh"])
    orig_mesh = _load_mesh_force_mesh(orig_mesh_path)
    tex_path = Path(conv.get("texture", {}).get("texture_path","")) if "texture" in conv else Path()
    orig_tex = Image.open(tex_path).convert("RGBA") if tex_path.exists() else Image.new("RGBA",(1024,1024),(128,128,128,255))

    # part -> original faces
    part_faces: Dict[str, List[int]] = {k: v.get("original_face_indices", []) for k,v in conv.get("parts", {}).items()}

    # final scene
    final_scene = trimesh.Scene()
    meta_parts = {}

    for pname, pinfo in sorted(comp.get("completed_parts", {}).items()):
        comp_path = Path(pinfo["completed_mesh_path"])
        if not comp_path.exists(): continue
        face_idx = np.array(part_faces.get(pname, []), np.int64)
        if face_idx.size == 0: continue

        ref_part = _submesh_by_faces(orig_mesh, face_idx)
        if getattr(ref_part.visual, "uv", None) is None:
            print(f"[WARN] {pname}: original part has no UV; skip."); continue

        comp_part = _load_mesh_force_mesh(comp_path)
        T = _icp_align(comp_part, ref_part, args.icp_samples, args.icp_ratio, args.icp_max_iter)
        comp_aligned = comp_part.copy(); comp_aligned.apply_transform(T)

        covered_mask, covered_src_tri = _face_vote_covered(
            comp_aligned, ref_part, args.eps_ratio, args.cos_threshold, vote=args.vote
        )

        faces = np.asarray(comp_aligned.faces, np.int64); verts = np.asarray(comp_aligned.vertices, np.float32)

        # ---- covered primitive (uses original texture)
        Vc, Fc, UVc = [], [], []
        vbase = 0
        for fi in np.where(covered_mask)[0]:
            vids = faces[fi]; pts3 = verts[vids]
            uvs = _uv_from_src_face(ref_part, int(covered_src_tri[fi]), pts3)
            for k in range(3):
                Vc.append(pts3[k]); UVc.append(uvs[k])
            Fc.append([vbase, vbase+1, vbase+2]); vbase += 3

        covered_mesh = None
        if len(Fc) > 0:
            covered_mesh = trimesh.Trimesh(vertices=np.asarray(Vc,np.float32),
                                           faces=np.asarray(Fc,np.uint32), process=False)
            covered_mesh.visual = trimesh.visual.texture.TextureVisuals(
                uv=np.asarray(UVc,np.float32), image=orig_tex
            )
            covered_mesh.visual.material = trimesh.visual.texture.SimpleMaterial(image=orig_tex)
            final_scene.add_geometry(covered_mesh, node_name=f"{pname}__covered")

            # overlays
            _save_uv_overlays(orig_tex,
                              np.asarray(UVc,np.float32),
                              np.asarray(Fc,np.int64),
                              parts_out / f"{pname}_covered_uv_overlay.png",
                              parts_out / f"{pname}_covered_coverage.png",
                              f"{pname} covered (orig)")
        # ---- new primitive (uses new gray texture)
        new_ids = np.where(~covered_mask)[0]
        new_mesh = None; new_tex_path = None
        if new_ids.size > 0:
            V1, F1, UV1 = _unwrap_subset(comp_aligned, new_ids)
            new_img = Image.new("RGBA", (orig_tex.width, orig_tex.height), (128,128,128,255))
            new_tex_path = parts_out / f"{pname}_new_texture.png"
            new_img.save(new_tex_path)

            new_mesh = trimesh.Trimesh(vertices=V1, faces=F1, process=False)
            new_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=UV1, image=new_img)
            new_mesh.visual.material = trimesh.visual.texture.SimpleMaterial(image=new_img)
            final_scene.add_geometry(new_mesh, node_name=f"{pname}__new")

            _save_uv_overlays(new_img,
                              np.asarray(UV1,np.float32),
                              np.asarray(F1,np.int64),
                              parts_out / f"{pname}_new_uv_overlay.png",
                              parts_out / f"{pname}_new_coverage.png",
                              f"{pname} new (xatlas)")

        meta_parts[pname] = {
            "faces_total": int(len(comp_aligned.faces)),
            "faces_covered": int(covered_mask.sum()),
            "faces_new": int((~covered_mask).sum()),
            "new_texture": str(new_tex_path) if new_ids.size>0 else None
        }

    # single GLB
    final_glb = outdir / "completed_textured_scene.glb"
    final_scene.export(str(final_glb))

    # metadata
    meta = {
        "inputs": {
            "original_mesh": str(orig_mesh_path),
            "original_texture": str(tex_path) if tex_path.exists() else None,
            "step1_dir": str(step1), "step2_dir": str(step2)
        },
        "outputs": {"scene_glb": str(final_glb), "parts_dir": str(parts_out)},
        "params": {
            "eps_ratio": args.eps_ratio, "cos_threshold": args.cos_threshold,
            "vote": bool(args.vote),
            "icp_samples": args.icp_samples, "icp_ratio": args.icp_ratio, "icp_max_iter": args.icp_max_iter
        },
        "parts": meta_parts
    }
    with open(outdir / "texture_transfer_metadata.json","w") as f:
        json.dump(meta,f,indent=2)

    print(f"[OK] single GLB: {final_glb}")
    print(f"[OK] new textures & overlays in: {parts_out}")

if __name__ == "__main__":
    main()
