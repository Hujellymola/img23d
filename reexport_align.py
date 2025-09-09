import bpy, sys, os, math, argparse
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Reorient a GLB and reexport")
    p.add_argument("--in", dest="inp", required=True, help="Input .glb/.gltf path")
    p.add_argument("--out", dest="out", required=True, help="Output .glb path")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--z-to-y", action="store_true", help="Rotate -90 deg about X (Z axis to +Y)")
    g.add_argument("--y-to-z", action="store_true", help="Rotate +90 deg about X (+Y to +Z)")
    return p.parse_args(argv)


def import_model(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def get_mesh_object():
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        return None
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.active_object


def clear_parent(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    except Exception:
        pass


def apply_all_transforms(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def rotate_for_alignment(obj, z_to_y=False, y_to_z=False):
    # Using right-hand rotations about X
    if z_to_y:
        # Map local/world +Z to +Y: rotate -90 deg about X
        rx = -90.0
    elif y_to_z:
        # Map +Y to +Z: rotate +90 deg about X
        rx = 90.0
    else:
        return
    obj.rotation_euler = (
        math.radians(rx),
        0.0,
        0.0,
    )
    apply_all_transforms(obj)


def export_glb(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=False,
        # Keep defaults for axis since Blender glTF exporter handles Y-up, -Z forward.
    )


def main():
    args = parse_args()
    inp = os.path.abspath(os.path.expanduser(args.inp))
    outp = os.path.abspath(os.path.expanduser(args.out))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(inp)

    obj = get_mesh_object()
    if obj is None:
        raise RuntimeError("No mesh found after import")

    clear_parent(obj)
    # Optional: set origin to bounds center so rotation is about the geometry center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    rotate_for_alignment(obj, z_to_y=args.z_to_y, y_to_z=args.y_to_z)

    export_glb(outp)
    print(f"Exported aligned model to: {outp}")


if __name__ == "__main__":
    main()

