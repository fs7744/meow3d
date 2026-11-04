"""用主项目的 Qwen-Image-2.1 生成角色的多视角图（投影管线的视图来源）。

**必须用主项目 venv 跑**（py3.13 / torch2.6 / diffusers0.41）：

    "D:/code/2d-game-meow/.venv/Scripts/python.exe" tools/qwen_views.py \
        --ref out/in/knight_matte.png --out-dir .tmp/qwen_views --views 4

主项目目录**严格只读**：
  · PYTHONDONTWRITEBYTECODE=1，不写 __pycache__
  · sys.path 只插入主项目根用于 import，cwd 始终在 meow3d
  · 所有输出写 meow3d/.tmp 与 meow3d/out
  · 权重通过 tools/link_qwen.py 硬链接到 meow3d/models/qwen（源 inode 不变）

生成结果给 meow3d 的 DiffusionViewProvider 消费（py3.12 环境）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# **必须让 offload 保持开启**：24G 卡上关掉它会「显存满、一直转、不报错」。
# 0 只适合 40G+ 的卡（每张省 1~2s 搬运）。
os.environ.setdefault("MEOW_QWEN_OFFLOAD", "1")

ROOT = Path(__file__).resolve().parent.parent
HOST = Path(r"D:\code\2d-game-meow")
sys.path.insert(0, str(HOST))          # 只读：仅用于 import workflows.engine

from PIL import Image                  # noqa: E402
import torch                           # noqa: E402
from workflows.engine import Engine    # noqa: E402

MODELS = ROOT / "models" / "qwen"

VIEW_PROMPTS = [
    "Show the SAME character from the front, in the exact same T-pose with arms "
    "stretched horizontally, full body from head to feet. Keep the identical armor, "
    "colors, crest and cape. Plain light gray background, even lighting.",
    "Show the SAME character rotated 45 degrees, three-quarter right view, "
    "arms still in T-pose, full body. Keep the identical armor, colors, crest and cape. "
    "Plain light gray background, even lighting.",
    "Show the SAME character from the exact RIGHT SIDE profile, arms in T-pose "
    "(one arm pointing toward the viewer), full body. Keep the identical armor, "
    "colors, crest and cape. Plain light gray background, even lighting.",
    "Show the SAME character turned 135 degrees, three-quarter back-right view, "
    "arms still in T-pose, full body. Keep the identical armor, colors, crest and cape. "
    "Plain light gray background, even lighting.",
    "Show the SAME character from DIRECTLY BEHIND, back view, arms in T-pose, "
    "full body. Keep the identical armor, colors, crest and cape. "
    "Plain light gray background, even lighting.",
    "Show the SAME character turned 225 degrees, three-quarter back-left view, "
    "arms in T-pose, full body. Keep the identical armor, colors, crest and cape. "
    "Plain light gray background, even lighting.",
    "Show the SAME character from the exact LEFT SIDE profile, arms in T-pose, "
    "full body. Keep the identical armor, colors, crest and cape. "
    "Plain light gray background, even lighting.",
    "Show the SAME character turned 315 degrees, three-quarter front-left view, "
    "arms in T-pose, full body. Keep the identical armor, colors, crest and cape. "
    "Plain light gray background, even lighting.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(ROOT / "out/in/knight_matte.png"))
    ap.add_argument("--out-dir", default=str(ROOT / ".tmp/qwen_views"))
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--indices", default="0,2,4,6",
                    help="要生成的相机序号（对应 make_cameras 的方位角）；"
                         "默认 4 个正交视角 0/2/4/6 = 正/右/背/左")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="只生成 1 张，验证管线能跑")
    args = ap.parse_args()

    ref = Image.open(args.ref).convert("RGBA")
    # 透明底合成浅灰，避免 Qwen 把透明区当噪声
    bg = Image.new("RGB", ref.size, (200, 202, 206))
    bg.paste(ref, mask=ref.getchannel("A"))
    ref = bg
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    indices = [0] if args.smoke else [
        int(x) for x in args.indices.split(",") if x.strip()][:args.views]
    size = args.size // 32 * 32

    print(f"加载 Qwen（权重 {MODELS}）…", flush=True)
    t0 = time.time()
    eng = Engine(str(MODELS))
    pipe = eng.txt2img_pipe("qwen-image-2.1")
    print(f"管线加载 {time.time() - t0:.1f}s", flush=True)

    # Lightning 少步 LoRA：8 步，画质几乎不掉
    try:
        eng.apply_loras(pipe, [{"name": "qwen21_lightning", "scale": 1.0}], scope="temp")
        print("已挂 Lightning LoRA", flush=True)
    except Exception as e:  # noqa: BLE001
        print("Lightning 挂载失败（不影响跑）:", e, flush=True)

    pipe.set_progress_bar_config(disable=True)
    for k, i in enumerate(indices):
        prompt = VIEW_PROMPTS[i]
        gen = torch.Generator("cuda").manual_seed(args.seed + i * 1000)
        kw = dict(
            prompt=prompt,
            image=ref.resize((size, size), Image.LANCZOS),
            width=size, height=size,
            num_inference_steps=max(8, min(args.steps, 60)),
            true_cfg_scale=1.0,
            output_resolution=size,
            generator=gen,
        )
        t = time.time()
        out = pipe(**kw)
        im = out.images[0].convert("RGB")
        p = outdir / f"view_{i:02d}.png"
        im.save(p)
        print(f"  [{i:02d}] {p.name}  {time.time() - t:.1f}s", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    print("完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
