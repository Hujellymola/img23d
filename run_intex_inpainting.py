#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Run InTeX on Step 3 outputs (rebaked per-part GLBs) to inpaint black regions only.

Features
--------
1) Batch mode (recommended): read Step 3's texture_transfer_metadata.json, iterate every part's
   `<part>_rebaked.glb`, call InTeX `main.py` to inpaint, then assemble a single final GLB.
2) Single-mesh mode: take any one GLB with UV + texture, run InTeX once (useful for debugging).
3) Rich terminal prints: per-part timing, counts, and a final summary.
4) Outputs:
   - <out>/parts_intex/<part>_inpainted.glb  (per-part, InTeX output)
   - <out>/final_textured_model.glb          (assembled scene of all inpainted parts)
   - <out>/intex_run_metadata.json           (summary metadata)

Assumptions
-----------
- Step 3 produced _rebaked.glb per part with a single material and one rebaked texture.
- New (untextured) regions were filled with PURE BLACK (0,0,0), to act as InTeX inpainting mask.
- InTeX CLI (Hydra style) is available next to this repo, e.g. ../InTeX/main.py

InTeX CLI (from official README):
  python main.py --config configs/revani.yaml \
                 mesh=data/dragon.glb \
                 prompt="a red pet dragon with fire patterns" \
                 save_path=dragon_fire.glb \
                 text_dir=True
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import trimesh  # only used for assembling final result


# -----------------------------
# Helpers
# -----------------------------

def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _save_json(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def _check_intex_repo(repo_dir: Path) -> Path:
    """Validate InTeX repo path and return main.py path."""
    if not repo_dir.exists():
        raise FileNotFoundError(f"InTeX repo not found: {repo_dir}")
    main_py = repo_dir / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Cannot find InTeX main.py in {repo_dir}")
    return main_py


def _run_intex(
    intex_main_py: Path,
    mesh_path: Path,
    save_path: Path,
    prompt: str,
    config_rel: str,
    extra_args: Optional[List[str]] = None,
    python_exec: str = sys.executable,
) -> None:
    """
    Call InTeX CLI with hydra-style arguments.

    Args:
        intex_main_py: path to InTeX/main.py
        mesh_path: input mesh (GLB) with UV+texture; black regions = inpaint mask
        save_path: output mesh path (GLB)
        prompt: text prompt to guide inpainting
        config_rel: e.g., 'configs/revani.yaml' (relative to InTeX repo)
        extra_args: optional extra hydra args, e.g. ["iters=30", "seed=0"]
        python_exec: python executable (defaults to current interpreter)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exec,
        str(intex_main_py),
        "--config", config_rel,
        f"mesh={str(mesh_path)}",
        f"prompt={prompt}",
        f"save_path={str(save_path)}",
        "text_dir=True",
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Run from InTeX repo dir to keep relative paths (configs/...) working.
    print(f"    [InTeX] CMD: {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(intex_main_py.parent), check=True)
    dt = time.perf_counter() - t0
    print(f"    [InTeX] Done in {dt:.1f}s -> {save_path}")


def _assemble_scene(part_glbs: List[Path], out_glb: Path) -> None:
    """Load per-part GLBs and assemble into one scene (each part becomes one node)."""
    scene = trimesh.Scene()
    for p in part_glbs:
        try:
            m = trimesh.load(str(p), force="scene")  # InTeX writes a single-mesh GLB; load scene for safety
            # Flatten scene -> geometries; keep node names if available
            if isinstance(m, trimesh.Scene):
                # use file stem as fallback node name
                node_name = p.stem
                scene.add_geometry(m.dump(concatenate=True), node_name=node_name)
            else:
                m.name = p.stem
                scene.add_geometry(m, node_name=m.name)
        except Exception as e:
            print(f"    [WARN] Failed to load {p}: {e}")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_glb))
    print(f"[OK] Assembled scene -> {out_glb}")


# -----------------------------
# Core runners
# -----------------------------

def run_batch_from_step3(
    step3_dir: Path,
    intex_repo: Path,
    config_rel: str,
    prompt: str,
    out_dir: Path,
    only_parts: Optional[List[str]] = None,
    skip_if_exists: bool = True,
    extra_hydra: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Batch mode: iterate parts from Step 3 metadata and inpaint each rebaked GLB.

    Returns metadata dict for the run (per-part timing & outputs).
    """
    meta_path = step3_dir / "texture_transfer_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Step 3 metadata JSON: {meta_path}")
    step3_meta = _load_json(meta_path)

    intex_main_py = _check_intex_repo(intex_repo)

    parts_info = step3_meta.get("parts", {})
    part_items = sorted(parts_info.items())  # [(part_name, info), ...]

    out_parts_dir = out_dir / "parts_intex"
    out_parts_dir.mkdir(parents=True, exist_ok=True)

    run_meta: Dict[str, Any] = {
        "mode": "batch",
        "intex_repo": str(intex_repo),
        "config": config_rel,
        "prompt": prompt,
        "step3_dir": str(step3_dir),
        "outputs": {"parts_dir": str(out_parts_dir)},
        "parts": {},
        "extra_hydra": extra_hydra or [],
    }

    print(f"[START] Batch InTeX on Step 3 outputs")
    print(f"  - Step3 dir: {step3_dir}")
    print(f"  - InTeX repo: {intex_repo}")
    print(f"  - Config: {config_rel}")
    print(f"  - Prompt: {prompt}")
    if only_parts:
        print(f"  - Only parts: {only_parts}")
    if extra_hydra:
        print(f"  - Extra hydra args: {extra_hydra}")

    total_t0 = time.perf_counter()
    produced_paths: List[Path] = []

    for idx, (part_name, info) in enumerate(part_items, start=1):
        if only_parts and part_name not in only_parts:
            continue

        rebaked_glb = info.get("rebaked_glb")
        if not rebaked_glb or not Path(rebaked_glb).exists():
            print(f"[{idx:02d}] {part_name}  SKIP (no rebaked_glb)")
            continue

        # Optional: skip parts that have no 'new' faces
        faces_new = info.get("faces_new", 0)
        if faces_new == 0:
            print(f"[{idx:02d}] {part_name}  SKIP (no new faces)")
            continue

        out_glb = out_parts_dir / f"{part_name}_inpainted.glb"
        if skip_if_exists and out_glb.exists():
            print(f"[{idx:02d}] {part_name}  SKIP (exists) -> {out_glb}")
            produced_paths.append(out_glb)
            run_meta["parts"][part_name] = {"input": rebaked_glb, "output": str(out_glb), "skipped": True}
            continue

        print(f"[{idx:02d}] {part_name}  RUN")
        print(f"    input:  {rebaked_glb}")
        print(f"    output: {out_glb}")

        t0 = time.perf_counter()
        try:
            _run_intex(
                intex_main_py=intex_main_py,
                mesh_path=Path(rebaked_glb),
                save_path=out_glb,
                prompt=prompt,
                config_rel=config_rel,
                extra_args=extra_hydra,
            )
            ok = True
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] InTeX failed for {part_name}: {e}")
            ok = False
        dt = time.perf_counter() - t0

        run_meta["parts"][part_name] = {
            "input": rebaked_glb,
            "output": str(out_glb),
            "ok": ok,
            "time_sec": dt,
            "faces_new": faces_new,
        }
        if ok:
            produced_paths.append(out_glb)

    total_dt = time.perf_counter() - total_t0
    print(f"[DONE] Batch InTeX, elapsed {total_dt:.1f}s")

    # Assemble final GLB if we have outputs
    if produced_paths:
        final_glb = out_dir / "final_textured_model.glb"
        _assemble_scene(produced_paths, final_glb)
        run_meta["outputs"]["final_scene"] = str(final_glb)
    else:
        print("[WARN] No parts were processed; skip assembling final scene.")

    return run_meta


def run_single_mesh(
    mesh_path: Path,
    intex_repo: Path,
    config_rel: str,
    prompt: str,
    out_dir: Path,
    extra_hydra: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Single-mesh mode: run InTeX once on a given GLB (with UV+texture).
    """
    intex_main_py = _check_intex_repo(intex_repo)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_glb = out_dir / (mesh_path.stem + "_inpainted.glb")

    print(f"[START] Single-mesh InTeX")
    print(f"  - Mesh:   {mesh_path}")
    print(f"  - InTeX:  {intex_repo}")
    print(f"  - Config: {config_rel}")
    print(f"  - Prompt: {prompt}")
    if extra_hydra:
        print(f"  - Extra hydra args: {extra_hydra}")

    try:
        _run_intex(
            intex_main_py=intex_main_py,
            mesh_path=mesh_path,
            save_path=out_glb,
            prompt=prompt,
            config_rel=config_rel,
            extra_args=extra_hydra,
        )
        ok = True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] InTeX failed: {e}")
        ok = False

    run_meta = {
        "mode": "single",
        "intex_repo": str(intex_repo),
        "config": config_rel,
        "prompt": prompt,
        "input_mesh": str(mesh_path),
        "output_mesh": str(out_glb),
        "ok": ok,
        "extra_hydra": extra_hydra or [],
    }
    return run_meta


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: Run InTeX on Step 3 outputs (per-part inpainting)")
    # Batch mode (from Step 3)
    p.add_argument("--step3-dir", type=str, default="output/step3_tex_transfer_1",
                   help="Directory of Step 3 outputs (must contain texture_transfer_metadata.json)")
    # Single-mesh mode
    p.add_argument("--single-mesh", type=str, default=None,
                   help="Path to a single GLB with UV+texture to run InTeX on")

    # Common
    p.add_argument("--intex-repo", type=str, default="../dependencies/InTeX",
                   help="Path to InTeX/ repo (must contain main.py)")
    p.add_argument("--config", type=str, default="configs/revani.yaml",
                   help="Config path relative to InTeX repo (e.g., configs/revani.yaml)")
    p.add_argument("--prompt", type=str, default="seamlessly extend the existing material",
                   help="Text prompt guiding the inpainting")
    p.add_argument("--output-dir", type=str, default="output/step4_intex",
                   help="Output directory")
    p.add_argument("--only-parts", type=str, nargs="*", default=None,
                   help="If set, only run these part names (batch mode)")
    p.add_argument("--no-skip", action="store_true",
                   help="Do not skip parts if output already exists")
    p.add_argument("--extra-hydra", type=str, nargs="*", default=None,
                   help="Extra hydra args passed to InTeX (e.g., iters=30 seed=0)")

    args = p.parse_args()

    if (args.step3_dir is None) == (args.single_mesh is None):
        p.error("Please specify exactly ONE of: --step3-dir  OR  --single-mesh")

    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.step3_dir is not None:
        run_meta = run_batch_from_step3(
            step3_dir=Path(args.step3_dir),
            intex_repo=Path(args.intex_repo),
            config_rel=args.config,
            prompt=args.prompt,
            out_dir=out_dir,
            only_parts=args.only_parts,
            skip_if_exists=(not args.no_skip),
            extra_hydra=args.extra_hydra,
        )
    else:
        run_meta = run_single_mesh(
            mesh_path=Path(args.single_mesh),
            intex_repo=Path(args.intex_repo),
            config_rel=args.config,
            prompt=args.prompt,
            out_dir=out_dir,
            extra_hydra=args.extra_hydra,
        )

    # Save run metadata
    meta_path = out_dir / "intex_run_metadata.json"
    _save_json(run_meta, meta_path)
    print(f"[OK] Saved run metadata -> {meta_path}")


if __name__ == "__main__":
    main()