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
    R, E, O = 1.0, 1.0, 0.0    # radius, elevation, look_offset
    CS, CE  = 0.1, 10.0         # clip start/end
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
        # bpy.ops.import_scene.obj(
        #     filepath=MODEL,
        #     axis_forward='-Z',
        #     axis_up='Y',
        #     use_split_objects=True,
        #     use_split_groups=True,
        #     use_image_search=True,
        # )
    else:
        raise ValueError(f"Unsupported model suffix: {ext}")

    # objs = [o for o in bpy.data.objects if o.type == 'MESH']
    # print("[INFO] Imported mesh objects:", [o.name for o in objs])

    # new_meshes = [o for o in bpy.data.objects if o.type=='MESH' and o.name not in pre_objs]
    # print(f"[INFO] Imported mesh objects: {[o.name for o in new_meshes]}")
    # obj = new_meshes[0]  # 你这份就是单网格

    # # 三角化
    # bpy.context.view_layer.objects.active = obj
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='SELECT')
    # bpy.ops.mesh.quads_convert_to_tris()   # 把四边面转成三角
    # bpy.ops.object.mode_set(mode='OBJECT')

    # # 按材质拆分为多个对象
    # bpy.context.view_layer.objects.active = obj
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='SELECT')
    # bpy.ops.mesh.separate(type='MATERIAL')  # 新建若干对象，每个对象只含一种材质
    # bpy.ops.object.mode_set(mode='OBJECT')

    # # 统计每个对象的面数（应接近你 OBJ 注释的 1754/1620/154/22/22）
    # objs = [o for o in bpy.context.scene.objects if o.type=='MESH']
    # grand = collections.Counter()
    # for o in objs:
    #     mats = o.data.materials
    #     # 现在每个 o 通常只有 1 个材质
    #     counts = collections.Counter(mats[p.material_index].name for p in o.data.polygons)
    #     total = len(o.data.polygons)
    #     print(f"[OBJ] {o.name}: tris={total}  mats={list(counts.items())}")
    #     grand.update(counts)

    # print("\n[TOTAL]")
    # for k,v in grand.items():
    #     print(f"  {k}: {v}")

    # grand = collections.Counter()
    # tris_total = 0
    # for o in objs:
    #     me = o.data
    #     # 统计三角数（确保与 3572 对齐）
    #     tris = sum(1 if len(me.polygons[i].vertices)==3 else 0 for i in range(len(me.polygons)))
    #     tris_total += tris

    #     mats = me.materials
    #     counts = collections.Counter(mats[p.material_index].name for p in me.polygons)
    #     print(f"\n[OBJ] {o.name}: polys={len(me.polygons)} tris(est)={tris}")
    #     for k,v in counts.items():
    #         print(f"  - {k}: {v}")

    #     grand.update(counts)

    # print("\n[TOTAL]")
    # print("  polys sum:", sum(grand.values()))
    # print("  tris  sum:", tris_total)
    # for k,v in grand.items():
    #     print(f"  {k}: {v}")

    # # 这里是假设你已经把它放到了导入之前
    # new_meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.name not in pre_objs]
    # print(f"[INFO] Imported mesh objects: {[o.name for o in new_meshes]}")

    # # --- 4) 逐对象统计“每个材质覆盖了多少面” ---
    # grand = collections.Counter()
    # for obj in new_meshes:
    #     mats = obj.data.materials
    #     counts = collections.Counter(mats[p.material_index].name for p in obj.data.polygons)
    #     total = len(obj.data.polygons)
    #     print(f"\n[OBJ] {obj.name}: faces={total}")
    #     for name, cnt in counts.items():
    #         print(f"  - {name}: {cnt}/{total} ({cnt/total:.1%})")
    #     grand.update(counts)

    # # --- 5) 汇总 & 断言关键材质是否被使用 ---
    # print("\n[TOTAL]")
    # total_faces = sum(grand.values())
    # for name, cnt in grand.items():
    #     print(f"  {name}: {cnt} ({cnt/total_faces:.1%})")

    # # 需要检查的材质名（按你的 .mtl）
    # targets = ["material_2_1_8", "material_4_2_8"]
    # for t in targets:
    #     used = grand.get(t, 0)
    #     print(f"[CHECK] {t} used faces = {used}")
    #     # 如果想在检测失败时直接报错，可启用下一行：
    #     # assert used > 0, f"{t} 未被任何面使用"


    # # 如果没提前保存 pre_objs，就退而求其次：取场景里所有 mesh
    # if not new_meshes:
    #     new_meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']

    # print(f"[INFO] Imported mesh objects: {[o.name for o in new_meshes]}")

    # # 逐对象统计材质覆盖
    # grand_counts = collections.Counter()
    # for o in new_meshes:
    #     if not o.data or not o.data.materials: 
    #         continue
    #     mats = o.data.materials
    #     local = collections.Counter(mats[p.material_index].name for p in o.data.polygons)
    #     total = len(o.data.polygons)
    #     print(f"[OBJ] {o.name}: faces={total}, mats={list(set(mats))}")
    #     for k,v in local.items():
    #         print(f"  - {k}: {v}/{total} ({v/total:.1%})")
    #     grand_counts.update(local)

    # # 全局汇总
    # total_faces = sum(grand_counts.values())
    # print("[TOTAL] faces:", total_faces)
    # for k,v in grand_counts.items():
    #     print(f"  {k}: {v} ({v/total_faces:.1%})")

    # # 顺便打印一下材质里是否有贴图节点、贴图路径是否存在
    # for mat in bpy.data.materials:
    #     has_img = False; img_path=None; exists=False
    #     if mat.use_nodes and mat.node_tree:
    #         for n in mat.node_tree.nodes:
    #             if n.type == 'TEX_IMAGE' and n.image and n.image.filepath:
    #                 has_img = True
    #                 img_path = bpy.path.abspath(n.image.filepath)
    #                 exists = os.path.exists(img_path)
    #                 break
    #     print(f"[MAT] {mat.name:20s} | TEX_NODE={has_img} | IMG={img_path} | EXISTS={exists}")

    # kd_map = parse_mtl_kd(MTL_PATH)

    # for mat in bpy.data.materials:
    #     # 确保节点材质与 Principled
    #     if not mat.use_nodes:
    #         mat.use_nodes = True
    #     nt = mat.node_tree
    #     nodes = nt.nodes
    #     bsdf = next((n for n in nodes if n.type=='BSDF_PRINCIPLED'), None)
    #     if not bsdf:
    #         bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    #         out = next((n for n in nodes if n.type=='OUTPUT_MATERIAL'), None) or nodes.new("ShaderNodeOutputMaterial")
    #         nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    #     # 优先寻找贴图
    #     tex_nodes = [n for n in nodes if n.type=='TEX_IMAGE' and n.image and n.image.filepath]
    #     linked = False
    #     for t in tex_nodes:
    #         path = bpy.path.abspath(t.image.filepath)
    #         if os.path.exists(path):
    #             nt.links.new(t.outputs.get('Color'), bsdf.inputs.get('Base Color'))
    #             linked = True
    #             break

    #     # 没贴图 → 用 .mtl 的 Kd 兜底（若没有就给个中性灰）
    #     if not linked:
    #         if mat.name in kd_map:
    #             r,g,b = kd_map[mat.name]
    #             bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    #         else:
    #             bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)

    # # # 2) 合并网格（只有>1个时）
    # # objs = [o for o in bpy.context.scene.objects if o.type=='MESH']
    # # bpy.ops.object.select_all(action='DESELECT')
    # # for o in objs: o.select_set(True)
    # # if objs:
    # #     bpy.context.view_layer.objects.active = objs[0]
    # # if len(objs) > 1:
    # #     bpy.ops.object.join()
    # obj = bpy.context.active_object

    # # 3) UV：若无 UV 自动展开，否则贴图也显示不出来
    # if obj and obj.type=='MESH':
    #     me = obj.data
    #     if not me.uv_layers:
    #         bpy.ops.object.mode_set(mode='EDIT')
    #         bpy.ops.mesh.select_all(action='SELECT')
    #         bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    #         bpy.ops.object.mode_set(mode='OBJECT')
    # debug_and_bind_materials()
    

    # for mat in bpy.data.materials:
    #     has_img = False; img_path = None; img_exists = False
    #     if mat.use_nodes and mat.node_tree:
    #         for n in mat.node_tree.nodes:
    #             if n.type == 'TEX_IMAGE' and n.image:
    #                 has_img = True
    #                 img_path = bpy.path.abspath(n.image.filepath) if n.image.filepath else None
    #                 img_exists = os.path.exists(img_path) if img_path else False
    #                 break
    #     print(f"[MAT] {mat.name:20s} | TEX_NODE={has_img} | IMG={img_path} | EXISTS={img_exists}")

    # # # —— 合并（如有必要）
    # # objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    # # bpy.ops.object.select_all(action='DESELECT')
    # # for o in objs: o.select_set(True)
    # # if objs:
    # #     bpy.context.view_layer.objects.active = objs[0]
    # # if len(objs) > 1:
    # #     bpy.ops.object.join()


    # === 等比缩放导入的模型，使其包围盒在 [-1,1]^3 中 ===
    objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    bpy.ops.object.join()  # 将所有 mesh 合并成一个对象
    obj = bpy.context.active_object

    # 计算包围盒范围
    # min_corner = np.min([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], axis=0)
    # max_corner = np.max([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], axis=0)

    # print("包围盒范围:", min_corner, max_corner)
    # scale = 2.0 / max((max_corner - min_corner))  # 缩放到最大边为 2

    # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    # 平移到原点、缩放到 [-1,1]^3
    # obj.location = (0, 0, 0)
    # obj.location = (0, 0, 1)
    # center = (min_corner + max_corner) / 2
    # obj.location = -center
    # bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
    # obj.scale *= scale
    # bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    # bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
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
    cam.data.clip_start = CS
    cam.data.clip_end   = CE
    scene.camera = cam

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,O))
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
    fov = cam.data.angle
    f = (RES/2)/math.tan(fov/2)
    intr = [[f,0,RES/2],[0,f,RES/2],[0,0,1]]
    params = {"intrinsic": intr, "views": {}}

    # # === 循环渲染 ===
    # for i in range(VIEWS):
    #     # 1) 设置帧号
    #     scene.frame_set(i)
    #     # 2) 布置相机环绕
    #     th = 2*math.pi*i/VIEWS
    #     cam.location = (R*math.sin(th), R*math.cos(th), E)
    #     # 3) 渲染（会同时输出 rgb_##.png 和 depth_##.exr）
    #     scene.render.image_settings.file_format = 'PNG'
    #     scene.render.filepath = os.path.join(OUT, f"rgb_{i:02d}.png")
    #     bpy.ops.render.render(write_still=True)
    #     print("Saved RGB:", f"rgb_{i:02d}.png")
    #     # 4) 读对应的 EXR
    #     exr = os.path.join(OUT, f"depth_{i:02d}.exr")
    #     img = bpy.data.images.load(exr, check_existing=True)
    #     flat = np.array(img.pixels[:]); bpy.data.images.remove(img)
    #     depth = flat[0::4].reshape((RES,RES))
    #     depth = np.flipud(depth)
    #     np.save(os.path.join(OUT, f"depth_{i:02d}.npy"), depth)
    #     vis = (depth - depth.min())/(depth.ptp()+1e-8)
    #     imageio.imwrite(os.path.join(OUT, f"depth_vis_{i:02d}.png"),
    #                     (vis*255).astype('uint8'))
    #     # 5) 存外参
    #     params["views"][f"{i:02d}"] = np.array(cam.matrix_world.inverted()).tolist()

    # # === 写 camera.json ===
    # with open(os.path.join(OUT, "camera.json"), "w") as f:
    #     json.dump(params, f, indent=2)

    # print("✅ 全部渲染完成，输出在", OUT)

    # # === 循环渲染 ===
    # # 改为从 cube 的 8 个顶点采样相机位置
    # cube_vertices = [
    #     (-1, -1, -1),
    #     (-1, -1,  1),
    #     (-1,  1, -1),
    #     (-1,  1,  1),
    #     ( 1, -1, -1),
    #     ( 1, -1,  1),
    #     ( 1,  1, -1),
    #     ( 1,  1,  1),
    # ]

    # for i in range(VIEWS):
    #     scene.frame_set(i)

    #     # === 设置相机位置为 cube 的 8 个角上的方向向量
    #     dx, dy, dz = cube_vertices[i]
    #     cam.location = (R * dx, R * dy, R * dz)
    #     cam.keyframe_insert(data_path="location")

    #     # === 注视目标点
    #     bpy.ops.object.constraint_add(type='TRACK_TO')
    #     cam.constraints["Track To"].target = empty
    #     cam.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    #     cam.constraints["Track To"].up_axis = 'UP_Y'

    #     # === 渲染 RGB ===
    #     scene.render.image_settings.file_format = 'PNG'
    #     scene.render.filepath = os.path.join(OUT, f"rgb_{i:02d}.png")
    #     bpy.ops.render.render(write_still=True)
    #     print("Saved RGB:", f"rgb_{i:02d}.png")

    #     # === 读 EXR 深度图 ===
    #     exr = os.path.join(OUT, f"depth_{i:02d}.exr")
    #     img = bpy.data.images.load(exr, check_existing=True)
    #     flat = np.array(img.pixels[:]); bpy.data.images.remove(img)
    #     depth = flat[0::4].reshape((RES,RES))
    #     depth = np.flipud(depth)
    #     np.save(os.path.join(OUT, f"depth_{i:02d}.npy"), depth)
    #     vis = (depth - depth.min())/(depth.ptp()+1e-8)
    #     imageio.imwrite(os.path.join(OUT, f"depth_vis_{i:02d}.png"),
    #                     (vis*255).astype('uint8'))

    #     # === 存外参
    #     params["views"][f"{i:02d}"] = np.array(cam.matrix_world.inverted()).tolist()

    # # === 写 camera.json ===
    # with open(os.path.join(OUT, "camera.json"), "w") as f:
    #     json.dump(params, f, indent=2)

    # print("✅ 全部渲染完成，输出在", OUT)
    # top_E = 2.0 - E 
    top_E = -E     
    heights = [E, top_E]  # 两层高度
    view_positions = []
    for z in heights:
        for i in range(4):
            theta = 2 * math.pi * i / 4  # 0, 90, 180, 270 度
            x = R * math.sin(theta)
            y = R * math.cos(theta)
            view_positions.append((x, y, z))

    # === 循环渲染 ===
    for i, (x, y, z) in enumerate(view_positions):  # === 修改点 2: 使用新的视角位置 ===
        # 1) 设置帧号
        scene.frame_set(i)

        # 2) 设置相机位置
        cam.location = (x, y, z)  # === 修改点 3: 使用自定义的 (x,y,z) 坐标 ===

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
