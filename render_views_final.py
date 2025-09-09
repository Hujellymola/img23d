# render_views_final_fixed_frames.py
import bpy, os, math, json, numpy as np, imageio
from mathutils import Vector
import sys
import collections

import argparse
import re


def parse_mtl_kd(mtl_path):
    """读取 .mtl，返回 {材质名: (r,g,b)}，范围 0..1"""
    if not os.path.exists(mtl_path): return {}
    kd = {}
    cur = None
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith('newmtl'):
                cur = s.split(maxsplit=1)[1]
            elif cur and s.lower().startswith('kd '):
                parts = s.split()
                try:
                    r, g, b = map(float, parts[1:4])
                    kd[cur] = (r, g, b)
                except:
                    pass
    return kd

def check_obj_mtllib(obj_path):
    ok, mtllib = False, None
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().lower().startswith("mtllib"):
                    mtllib = line.strip().split(maxsplit=1)[1]
                    ok = True
                    break
    except Exception as e:
        print(f"[OBJ] read error: {e}")
    if not ok:
        print("[OBJ] ⚠️ 没有找到 mtllib 行：Blender 可能不会加载 .mtl")
    else:
        print(f"[OBJ] mtllib => {mtllib}")
    return ok, mtllib

def fix_mtl_texture_paths(mtl_path, images_dir):
    """把 .mtl 里裸文件名改成指向 ../images/xxx.jpg 的相对路径；已有路径的不动。"""
    if not os.path.exists(mtl_path):
        print(f"[MTL] ⚠️ 不存在: {mtl_path}")
        return
    images_dir = os.path.abspath(images_dir)
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    keys = ("map_kd","map_ks","map_ka","map_d","map_bump","bump","map_ns","map_pr","map_pm","map_ps")
    pat = re.compile(rf"^({'|'.join(keys)})\s+(.*)\s*$", re.I)

    changed = False
    out = []
    for line in lines:
        m = pat.match(line.strip())
        if m:
            key, p = m.groups()
            # 若已有目录就保持；否则到 images_dir 下找同名文件
            if os.path.dirname(p):
                out.append(line)  # 保持
                continue
            cand = os.path.join(images_dir, p)
            if os.path.exists(cand):
                rel = os.path.relpath(cand, os.path.dirname(mtl_path))
                new_line = f"{key} {rel}\n"
                out.append(new_line)
                changed = True
                print(f"[MTL] {key}: {p} -> {rel}")
            else:
                out.append(line)  # 找不到就保持原样
                print(f"[MTL] ⚠️ 贴图不存在: {p}")
        else:
            out.append(line)

    if changed:
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.writelines(out)
        print("[MTL] ✅ 已更新相对路径")
    else:
        print("[MTL] 无需修改")
        
def debug_and_bind_materials():
    print("\n=== 材质诊断 ===")
    # 统计贴图资源是否存在
    for img in bpy.data.images:
        path = bpy.path.abspath(img.filepath) if img.filepath else ""
        exists = os.path.exists(path) if path else False
        print(f"[IMG] {img.name:30s} -> {path}  {'OK' if exists else 'MISSING'}")

    # 遍历材质并修复/连接节点
    for mat in bpy.data.materials:
        print(f"\n[MAT] {mat.name}")
        if not mat.use_nodes:
            print("  - 没有节点，创建 Principled + Output")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
            continue

        nt = mat.node_tree
        nodes = nt.nodes
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            print("  - 没有 Principled, 创建并连接输出")
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            out = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None) or nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        tex_nodes = [n for n in nodes if n.type == 'TEX_IMAGE']
        if not tex_nodes:
            print("  - 没找到 Image Texture 节点（可能 .mtl 没指定贴图）")
            continue

        # 找一个可用贴图并连接
        linked = False
        for t in tex_nodes:
            path = bpy.path.abspath(t.image.filepath) if t.image and t.image.filepath else ""
            if path and os.path.exists(path):
                nt.links.new(t.outputs.get("Color"), bsdf.inputs.get("Base Color"))
                print(f"  - 已连接贴图到 BaseColor: {os.path.basename(path)}")
                linked = True
                break
        if not linked:
            print("  - 贴图节点存在但文件缺失或未设置路径")
            
if __name__ == "__main__":
    # 只保留 Python 部分的参数（忽略 Blender 自带参数）
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []  # 没有参数

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args(argv)

    MODEL = os.path.expanduser(args.model)
    OUT   = os.path.expanduser(args.out)

    # === 用户参数 ===
    # MODEL   = os.path.expanduser("~/Downloads/mug_-_firefox.glb")
    # MODEL   = os.path.expanduser("~/Downloads/nescafe_mug_.glb")
    # OUT     = os.path.expanduser("~/UAD/output/new_cube_2")
    VIEWS   = 8
    # Initial placeholders; will be recomputed after normalization
    R, E, O = 1.0, 1.0, 0.0    # radius (xy), elevation (z), look_offset
    CS, CE  = 0.1, 10.0         # clip start/end (will be scaled from camera distance)
    SAMPLES = 64
    RES     = 448

    os.makedirs(OUT, exist_ok=True)
    # OBJ_PATH = MODEL
    # MTL_PATH = os.path.splitext(OBJ_PATH)[0] + ".mtl"
    # IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(OBJ_PATH), "..", "images"))
    # _ , _ = check_obj_mtllib(OBJ_PATH)
    # fix_mtl_texture_paths(MTL_PATH, IMAGES_DIR)

    # === 清空 & 导入 ===
    bpy.ops.wm.read_factory_settings(use_empty=True)
    pre_objs = set(bpy.data.objects)
    # bpy.ops.import_scene.gltf(filepath=MODEL)
    ext = os.path.splitext(MODEL)[1].lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=MODEL)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(
            filepath=MODEL,
            axis_forward='-Z',
            axis_up='Y',
            use_split_objects=False,
            use_split_groups=False,
            use_image_search=True,
        )
    else:
        raise ValueError(f"Unsupported model suffix: {ext}")


    # === Normalize model ===
    # 1) Merge all imported meshes into a single object
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
    if objs:
        bpy.ops.object.join()
    obj = bpy.context.active_object

    # 2) Clear parent transforms (glTF often adds empties) and set origin to bounds center
    try:
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    except Exception:
        pass
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # 3) Compute world-space bounding box and scale to a unit bounding sphere (radius=1)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in corners); max_x = max(c.x for c in corners)
    min_y = min(c.y for c in corners); max_y = max(c.y for c in corners)
    min_z = min(c.z for c in corners); max_z = max(c.z for c in corners)
    dx = max_x - min_x; dy = max_y - min_y; dz = max_z - min_z
    # Bounding sphere radius = half of bbox diagonal length
    diag = (dx**2 + dy**2 + dz**2) ** 0.5
    r_curr = max(diag * 0.5, 1e-6)
    target_r = 1.0
    s = target_r / r_curr
    obj.scale = (obj.scale[0] * s, obj.scale[1] * s, obj.scale[2] * s)
    bpy.ops.object.transform_apply(scale=True)
    # Recompute after scaling (for safety)
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_x = min(c.x for c in corners); max_x = max(c.x for c in corners)
    min_y = min(c.y for c in corners); max_y = max(c.y for c in corners)
    min_z = min(c.z for c in corners); max_z = max(c.z for c in corners)
    dx = max_x - min_x; dy = max_y - min_y; dz = max_z - min_z
    diag = (dx**2 + dy**2 + dz**2) ** 0.5
    r_norm = max(diag * 0.5, 1e-6)  # should be near 1.0
    # Compute world-space center for camera targeting and offsets
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    cz = (min_z + max_z) * 0.5
        
    # === 设置场景 ===
    scene = bpy.context.scene

    # === 把帧号从 0 开始 ===
    scene.frame_start   = 0
    scene.frame_end     = VIEWS - 1
    scene.frame_current = 0

    # === 背景、光源、相机同前 ===
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.5,0.5,0.5,1.0)
    bg.inputs[1].default_value = 1.0

    bpy.ops.object.light_add(type='SUN', location=(5,-5,5))
    bpy.context.object.data.energy = 10.0

    bpy.ops.object.camera_add(location=(0,-R,E), rotation=(math.radians(90),0,0))
    cam = bpy.context.object
    # Fix a reasonable FOV (vertical) and recompute a safe camera distance
    cam.data.angle = math.radians(60.0)
    fov = cam.data.angle  # vertical FOV (radians)
    # Distance needed to fit a sphere of radius r_norm within view: D >= r / tan(fov/2)
    safety = 1.15
    D = (r_norm / max(math.tan(fov/2), 1e-6)) * safety
    # Place initial camera at distance D on -Y axis, centered on the object
    cam.location = (cx, cy - D, cz)
    cam.data.clip_start = max(0.01, D * 0.05)
    cam.data.clip_end   = D * 10.0
    scene.camera = cam

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(cx, cy, cz + O))
    empty = bpy.context.object
    trk = cam.constraints.new("TRACK_TO")
    trk.target     = empty
    trk.track_axis = 'TRACK_NEGATIVE_Z'
    trk.up_axis    = 'UP_Y'

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.view_layers[0].use_pass_z = True

    # === 搭建节点 ===
    scene.use_nodes = True
    tree = scene.node_tree
    # 保留默认 RL→Composite
    rl = tree.nodes.get("Render Layers") or tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.get("Composite")   or tree.nodes.new("CompositorNodeComposite")
    if not any(l.from_node==rl and l.to_node==comp for l in tree.links):
        tree.links.new(rl.outputs["Image"], comp.inputs[0])
    # 追加 Depth 输出
    dout = tree.nodes.new("CompositorNodeOutputFile")
    dout.label        = "Depth"
    dout.base_path    = OUT
    dout.file_slots[0].path      = "depth_##"
    dout.format.file_format      = 'OPEN_EXR'
    dout.format.color_depth      = '32'
    tree.links.new(rl.outputs["Depth"], dout.inputs[0])

    # === 预计算内参 ===
    f = (RES/2)/math.tan(fov/2)
    intr = [[f,0,RES/2],[0,f,RES/2],[0,0,1]]
    params = {"intrinsic": intr, "views": {}}

  
    # Build 8 views at constant distance D: 4 around at z=+h*D and 4 at z=-h*D
    # Keep distance constant so framing is consistent and avoids cropping
    h = 0.25  # elevation fraction of D
    z_vals = [ D * h, -D * h ]
    view_positions = []
    for zv in z_vals:
        r_xy = max((D**2 - zv**2), 0.0) ** 0.5
        for i in range(4):
            theta = 2 * math.pi * i / 4
            x = cx + r_xy * math.sin(theta)
            y = cy + r_xy * math.cos(theta)
            view_positions.append((x, y, cz + zv))

    # === 循环渲染 ===
    for i, (x, y, z) in enumerate(view_positions):  # === 修改点 2: 使用新的视角位置 ===
        # 1) 设置帧号
        scene.frame_set(i)

        # 2) 设置相机位置
        cam.location = (x, y, z)

        # 3) 渲染（会同时输出 rgb_##.png 和 depth_##.exr）
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.join(OUT, f"rgb_{i:02d}.png")
        bpy.ops.render.render(write_still=True)
        print("Saved RGB:", f"rgb_{i:02d}.png")

        # 4) 读对应的 EXR
        exr = os.path.join(OUT, f"depth_{i:02d}.exr")
        img = bpy.data.images.load(exr, check_existing=True)
        flat = np.array(img.pixels[:]); bpy.data.images.remove(img)
        depth = flat[0::4].reshape((RES, RES))
        depth = np.flipud(depth)
        np.save(os.path.join(OUT, f"depth_{i:02d}.npy"), depth)
        # NumPy 2.0 removed ndarray.ptp(); use np.ptp(depth) instead
        vis = (depth - np.min(depth)) / (np.ptp(depth) + 1e-8)
        imageio.imwrite(os.path.join(OUT, f"depth_vis_{i:02d}.png"),
                        (vis * 255).astype('uint8'))

        # 5) 存外参
        params["views"][f"{i:02d}"] = np.array(cam.matrix_world.inverted()).tolist()

    # === 写 camera.json ===
    with open(os.path.join(OUT, "camera.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("✅ 全部渲染完成，输出在", OUT)
