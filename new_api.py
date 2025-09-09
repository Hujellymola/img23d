import os
import base64
import json
import argparse
import re
from openai import OpenAI

# ============ 参数解析 ============
parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path", type=str, required=True, help="Path to the single RGB image")
parser.add_argument("--overlay_root", type=str, required=True, help="Directory containing index_i/overlay_01.png")
parser.add_argument("--action", type=str, required=True, help="Action name (e.g., pour)")
parser.add_argument("--output_path", type=str, required=True, help="Where to save the result json file")
args = parser.parse_args()

rgb_path = os.path.expanduser(args.rgb_path)
overlay_root = os.path.expanduser(args.overlay_root)
action_name = args.action
output_path = os.path.expanduser(args.output_path)

# ============ 初始化 ============
client = OpenAI()

# ============ 编码图片 ============
def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# rgb_base64 = encode_image_base64(rgb_path)

# ============ 构造 prompt ============
# prompt = f"""You are given one RGB image and one highlighted region image for a mug.

# Your task: determine whether the highlighted region corresponds to the most relevant part for performing the action "{action_name}".

# When identifying regions, think about how the **action** works, for example, for the action **"pour"**:
# - You usually **hold the mug by the handle**.
# - Liquid exits through the **opening** or over the **rim**.

# Only answer with `True` or `False`, nothing else."""
prompt = f"""
You are given two images of a mug:
- One is a normal RGB image.
- One highlights a region on the mug (colored overlay).

You are also given an action: **"{action_name}"**.

Note: The current view may not fully show or even not show the highlighted region, 
because it is only one viewpoint of the object. 
Thus, the highlighted region might appear partially or be occluded from this angle.

First, think about **which parts of the mug are most important** for performing this action. For example:

- For action **"pour"**, the most important parts are:
  - The **opening** or **rim**, where the liquid exits;
  - The **handle**, which is used to hold the mug.

Then, look at the highlighted region and determine:  
Does it match **one of the most important parts** for this action?

Do NOT output your reasoning.

**Only return `True` or `False`, nothing else..**
"""


# ============ 遍历 index 目录并逐个询问 ============
index_dirs = sorted([
    d for d in os.listdir(overlay_root)
    if os.path.isdir(os.path.join(overlay_root, d)) and d.startswith("index_")
])

results = []

for index_name in index_dirs:
    # image_paths = []
    n_views = 8  # 假设有 8 个视角
    # # rgb 图片
    # for i in range(n_views):
    #     path = os.path.join(rgb_path, f"rgb_{i:02d}.png")
    #     if os.path.exists(path):
    #         image_paths.append(path)
    #     else:
    #         print(f"⚠️ Missing RGB image: {path}")
    
    # for i in range(n_views):
    #     overlay_path = os.path.join(overlay_root, index_name, f"overlay_{i:02d}.png")
    #     if os.path.exists(overlay_path):
    #         image_paths.append(overlay_path)
    #     else:
    #         print(f"⚠️ Missing overlay image: {overlay_path}")

    # overlay_path = os.path.join(overlay_root, index_name, "overlay_01.png")
    # if not os.path.exists(overlay_path):
    #     print(f"⚠️ Missing overlay image: {overlay_path}")
    #     results.append(False)
    #     continue

    # overlay_base64 = encode_image_base64(overlay_path)

    # messages = [{
    #     "role": "user",
    #     "content": [
    #         { "type": "text", "text": prompt },
    #         {
    #             "type": "image_url",
    #             "image_url": { "url": f"data:image/png;base64,{rgb_base64}" }
    #         },
    #         {
    #             "type": "image_url",
    #             "image_url": { "url": f"data:image/png;base64,{overlay_base64}" }
    #         }
    #     ]
    # }]
    # 构造消息
    content_list = [
        {
            "type": "text",
            "text": f"""
        You are given multiple images of a mug from different viewpoints.
        - The first {n_views} images are normal RGB images of the mug.
        - The next {n_views} images highlight the same region of the mug (from the same viewpoints).

        You are also given an action: **"{action_name}"**.

        Note: Each overlay corresponds to the same region, but from different angles.
        The current view may not fully show the highlighted region in every image.

        First, think about which parts of the mug are most important for performing this action.
        Then check: does the highlighted region (across all views) correspond to one of the important parts?

        Do NOT output your reasoning.

        **Only return `True` or `False`, nothing else..**
        If unsure, default to `False`.
        """
        }
    ]

    # 先添加 RGB 图（前 n_views 张）
    for i in range(n_views):
        rgb_img_path = os.path.join(rgb_path, f"rgb_{i:02d}.png")
        if os.path.exists(rgb_img_path):
            rgb_base64 = encode_image_base64(rgb_img_path)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{rgb_base64}"}
            })
        else:
            print(f"⚠️ Missing RGB image: {rgb_img_path}")

    # 再添加 overlay 图（后 n_views 张）
    for i in range(n_views):
        overlay_img_path = os.path.join(overlay_root, index_name, f"overlay_{i:02d}.png")
        if os.path.exists(overlay_img_path):
            overlay_base64 = encode_image_base64(overlay_img_path)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{overlay_base64}"}
            })
        else:
            print(f"⚠️ Missing overlay image: {overlay_img_path}")

    messages = [{
        "role": "user",
        "content": content_list
    }]

    # ==== 投票逻辑 ====
    vote_true = 0
    n_repeat = 3  # 重复次数
    for _ in range(n_repeat):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=20
            )
            reply = response.choices[0].message.content.strip().lower()

            if "true" in reply:
                vote_true += 1
            elif "false" not in reply:
                print(f"⚠️ Unexpected reply for {index_name}: {reply}")
        except Exception as e:
            print(f"❌ Error on {index_name}: {str(e)}")

    # 多数票决定最终结果
    final_result = (vote_true >= 2)  # 3 次里至少 2 次 True
    results.append(final_result)
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-4o",
    #         messages=messages,
    #         max_tokens=20
    #     )
    #     reply = response.choices[0].message.content.strip().lower()

    #     if "true" in reply:
    #         results.append(True)
    #     elif "false" in reply:
    #         results.append(False)
    #     else:
    #         print(f"⚠️ Unexpected reply for {index_name}: {reply}")
    #         results.append(False)
    # except Exception as e:
    #     print(f"❌ Error on {index_name}: {str(e)}")
    #     results.append(False)

print("length of index:", len(index_dirs))
print("length of results:", len(results))
print("🧠 GPT-4o Responses:", results)
# ============ 保存结果 ============
result_file = os.path.join(output_path, "result.json")
os.makedirs(output_path, exist_ok=True)

with open(result_file, "w") as f:
    json.dump(results, f)
