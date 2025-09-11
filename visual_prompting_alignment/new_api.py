import os
import base64
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ============ 参数解析 ============
parser = argparse.ArgumentParser()
parser.add_argument("--rgb_path", type=str, required=True, help="Path to the single RGB image")
parser.add_argument("--overlay_root", type=str, required=True, help="Directory containing index_i/overlay_01.png")
parser.add_argument("--action", type=str, required=True, help="Action name (e.g., pour)")
parser.add_argument("--role", type=str, default="actor", help="Role of the object part regarding the action (actor/recipient)")
parser.add_argument("--output_path", type=str, required=True, help="Where to save the result json file")
args = parser.parse_args()

rgb_path = os.path.expanduser(args.rgb_path)
overlay_root = os.path.expanduser(args.overlay_root)
action_name = args.action
role = args.role
output_path = os.path.expanduser(args.output_path)

print(f"action: {action_name}, role: {role}")

# ============ 初始化 ============
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise RuntimeError(f"OPENAI_API_KEY not found at {env_path}")

client = OpenAI(api_key=key)

# ============ 编码图片 ============
def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ============ 遍历 index 目录并逐个询问 ============
index_dirs = sorted([
    d for d in os.listdir(overlay_root)
    if os.path.isdir(os.path.join(overlay_root, d)) and d.startswith("index_")
])

results = []

for index_name in index_dirs:
    # image_paths = []
    n_views = 8  # 假设有 8 个视角

    # 构造消息
    content_list = [
        {
            "type": "text",
            "text": f"""
        You are given multiple images of an object from different viewpoints.
        - The first {n_views} images are normal posed RGB images of the object.
        - The next {n_views} images are the same posed images with a overlaying mask highlighting a specific region of the object.

        Note: Each overlay corresponds to the SAME 3D region, but from different angles.
        Due to occlusion, the current view may not fully show the highlighted region in every image.
        
        Your job is a strict binary decision for a given action and role: 
        Action: **"{action_name}"**
        Role: **"{role}"**.

        Definitions:
        - action: the physical operation (e.g., "pour", "grasp", "push", "cut", "twist").
        - role:
        - "actor": the object PART that directly produces the action’s primary effect
            (e.g., spout / pour lip / nozzle / mouth / rim opening emits liquid for "pour"; blade cuts for "cut").
        - "receiver": the object PART that is meant to receive or be acted upon
            (e.g., handle / dedicated gripping area for "grasp"; target opening/cavity for "pour" on the receiving object).

        Rules:
        1) Return True only if the highlighted region is one of the MOST DIRECTLY relevant parts
        for the given (action, role).
        2) If uncertain, return `False`.
        3) Output exactly one token: `True` or `False`. Do NOT output your reasoning.
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
    print(f"Index: {index_name}, Vote True: {vote_true}/{n_repeat}, Final: {final_result}")
    results.append(final_result)


print("length of index:", len(index_dirs))
print("length of results:", len(results))
print("🧠 GPT-4o Responses:", results)
# ============ 保存结果 ============
result_file = os.path.join(output_path, "result.json")
os.makedirs(output_path, exist_ok=True)

with open(result_file, "w") as f:
    json.dump(results, f)
