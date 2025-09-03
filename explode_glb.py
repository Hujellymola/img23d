# explode_glb.py
# Usage:
#   python explode_glb.py input.glb output.glb --radius 0.2 --y_offset 0.0
#   pip install pygltflib

import math
import argparse
from copy import deepcopy
from pygltflib import GLTF2

def attr_get(attrs, name):
    """Works for pygltflib.Attributes and dict-like cases."""
    try:
        return attrs.get(name)
    except Exception:
        return getattr(attrs, name, None)

def mesh_extent(gltf: GLTF2, mesh_index: int) -> float:
    """Estimate mesh size from POSITION accessor min/max."""
    if mesh_index is None:
        return 1.0
    m = gltf.meshes[mesh_index]
    for prim in m.primitives or []:
        pos_idx = attr_get(prim.attributes, "POSITION")
        if pos_idx is None:
            continue
        acc = gltf.accessors[pos_idx]
        if acc.min is None or acc.max is None:
            continue
        size = [abs(acc.max[i] - acc.min[i]) for i in range(3)]
        return max(size)
    return 1.0

def add_vec(a, b):
    a = a or [0.0, 0.0, 0.0]
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def gather_targets(gltf: GLTF2):
    """Prefer children of single root. Else use all roots."""
    scene_idx = gltf.scene if gltf.scene is not None else 0
    scene = gltf.scenes[scene_idx]
    roots = scene.nodes or []
    if len(roots) == 1:
        root = gltf.nodes[roots[0]]
        if root.children:
            return list(root.children)
    return list(roots)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--y_offset", type=float, default=0.0)
    args = ap.parse_args()

    gltf = GLTF2().load_binary(args.input)

    targets = gather_targets(gltf)
    if not targets:
        print("No nodes to explode.")
        return

    # precompute extents per mesh
    mesh_size = {}
    for ni in targets:
        node = gltf.nodes[ni]
        if node.mesh is not None and node.mesh not in mesh_size:
            mesh_size[node.mesh] = mesh_extent(gltf, node.mesh)

    n = len(targets)
    for k, ni in enumerate(targets):
        node = gltf.nodes[ni]
        theta = (2.0 * math.pi * k) / max(1, n)
        base = args.radius
        extra = 0.5 * mesh_size.get(node.mesh, 1.0)
        r = base + extra
        dx = r * math.cos(theta)
        dz = r * math.sin(theta)
        dy = args.y_offset
        node.translation = add_vec(node.translation, [dx, dy, dz])

    out = deepcopy(gltf)
    out.save_binary(args.output)
    print(f"Wrote: {args.output}")

if __name__ == "__main__":
    main()
