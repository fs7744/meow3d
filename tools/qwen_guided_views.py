"""用「同机位指引图」引导 Qwen 生成几何对齐的新视角（多相机投影的视图来源）。

与 `tools/qwen_views.py` 的差别只有一件事，但很关键：

| | qwen_views.py（旧） | 本脚本 |
|---|---|---|
| 输入 | 只有正面参考图 | 正面参考图 **+ 目标机位的结构图** |
| 角度控制 | 纯文字 prompt（"rotated 90 degrees"） | 深度图 / 素模渲染直接钉住轮廓 |
| 实测后果 | `view_02` 号称 90° 实际偏正面 | 侧视轮廓与 3D 逐像素对齐 |

三种模式：

* `beauty`  —— 把**模型自己在该机位的彩图**当底图做指令改图（几何 100% 正确，
              颜色/设计靠 Qwen 对齐参考图）。最稳，但依赖模型已有贴图。
* `control` —— ControlNet Union 的 depth 条件 + 参考图。完全不看模型贴图，
              适合白模路线。
* `both`    —— 深度条件 + 底图 + 参考图，三重约束（最慢，理论上最强）。

**必须用主项目 venv 跑**（py3.13 / torch2.6 / diffusers0.41）；主项目目录严格只读。

    "D:/code/2d-game-meow/.venv/Scripts/python.exe" tools/qwen_guided_views.py \
        --ref out/in/knight_matte.png --guide .tmp/guide_knight --mode control \
        --cameras 4 --indices 0,1,2,3 --out-dir .tmp/qwen_guided
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# **必须让 offload 保持开启**（主项目默认）：Qwen 峰值 16.4GB，而 transformer-fp8
# + text-encoder-fp8 常驻是 15.5GB，24G 卡上关掉 offload 会「显存满、一直转、
# 不报错」——实测单张 768² 卡 4 分钟以上。0 只适合 40G+ 的卡。
os.environ.setdefault("MEOW_QWEN_OFFLOAD", "1")

ROOT = Path(__file__).resolve().parent.parent
HOST = Path(r"D:\code\2d-game-meow")
sys.path.insert(0, str(HOST))          # 只读：仅用于 import workflows.*

from PIL import Image, ImageDraw     # noqa: E402
import torch                         # noqa: E402
from workflows.engine import Engine  # noqa: E402

MODELS = ROOT / "models" / "qwen"
CN_PATH = MODELS / "controlnet" / "qwen21_union.safetensors"

ANGLE_WORD = {0: "front view", 90: "right side profile view",
              180: "back view", 270: "left side profile view"}

PROMPT_EDIT = (
    "Two separate input images. The FIRST is a character design reference. "
    "The SECOND is a 3D render whose pose, camera angle and framing must be kept "
    "exactly as-is. "
    "CRITICAL POSE CONSTRAINT: the 3D mesh is in a STRICT T-POSE — BOTH arms are "
    "stretched straight out horizontally to the sides, perpendicular to the torso. "
    "This must be preserved in EVERY view. In side/profile views the arms overlap "
    "the body and appear as short horizontal bars: do NOT lower, bend, fold, hide "
    "or shorten the arms, and do NOT draw a relaxed standing pose. "
    "Output ONE single image containing ONLY the character from the "
    "second image, re-textured to match the design of the first image: same clothing "
    "and armor, same colors, same hair, same face. "
    "Do NOT put the two images side by side. No collage, no split screen, no "
    "before/after comparison, no duplicated figure. "
    "Full body from head to feet, even lighting, plain light gray background."
)

# 单图模式（默认）：只给彩图，不让模型看到第二张，从根上避免「拼两格」
PROMPT_EDIT_SINGLE = (
    "Clean up and refine this 3D character render into crisp finished game art. "
    "KEEP the exact same art style, the exact same color palette and brightness, "
    "and the exact same pose, camera angle and framing. Do not make it realistic "
    "or photographic, do not brighten it, do not change the hair color or the "
    "outfit colors. Only sharpen edges, remove texture seams and smudges, and add "
    "clean surface detail. One single character only, no collage, no split screen, "
    "no duplicated figure. Full body from head to feet, plain light gray background."
)

PROMPT_CONTROL = (
    "A full body game character in an exact T-pose with arms stretched horizontally, "
    "matching the silhouette and proportions of the depth map precisely. "
    "Following the reference image's design: same armor, same colors, same crest, "
    "same cape, same face. Detailed anime game character, clean sharp surfaces, "
    "even lighting, plain light gray background, head to feet visible."
)


def _open_rgba(p: str | Path) -> Image.Image:
    return Image.open(p).convert("RGBA")


def _flatten(im: Image.Image, bg=(200, 202, 206)) -> Image.Image:
    if im.mode != "RGBA":
        return im.convert("RGB")
    canvas = Image.new("RGB", im.size, bg)
    canvas.paste(im, mask=im.getchannel("A"))
    return canvas


def _fit(im: Image.Image, size: int) -> Image.Image:
    """等比塞进 size² 的浅灰画布（不拉伸，避免参考图比例被改）。"""
    im = im.convert("RGB")
    if im.size == (size, size):
        return im
    s = min(size / im.width, size / im.height)
    nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
    canvas = Image.new("RGB", (size, size), (200, 202, 206))
    canvas.paste(im.resize((nw, nh), Image.LANCZOS),
                 ((size - nw) // 2, (size - nh) // 2))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="角色参考图（正面，最好带 alpha）")
    ap.add_argument("--guide", required=True, help="tools/guidance.py 的输出目录")
    ap.add_argument("--mode", default="control",
                    choices=("beauty", "control", "both"))
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--indices", default="0,1,2,3")
    ap.add_argument("--plan", help="相机方案 JSON（tools/plan_cameras.py）；"
                                   "按它的方位角/俯仰角写提示词，不靠等分环推算")
    ap.add_argument("--guide-pass", default="beauty",
                    choices=("beauty", "clay", "normal", "depth_c"),
                    help="结构底图。白模 beauty 是纯白，侧视 T-pose 手臂几乎看不见；"
                         "clay 素模带明暗/cavity，几何轮廓更清楚")
    ap.add_argument("--two-images", action="store_true",
                    help="同时给参考图与彩图。**默认不给**：两张全身站姿图一起进来时，"
                         "Qwen 会时不时画成「左原图右结果」的两格拼图，飘带一重叠就连"
                         "自动切分都救不回来（实测汉服背面就是一例）")
    ap.add_argument("--out-dir", default=str(ROOT / ".tmp" / "qwen_guided"))
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.9,
                    help="ControlNet 强度（0=不听，1=完全照抄深度）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-lightning", action="store_true",
                    help="不挂 8 步蒸馏 LoRA（更慢更准，步数用 --steps）")
    ap.add_argument("--lora", action="append", default=[],
                    help="额外 LoRA，形如 名称:强度，可重复。例如 "
                         "VNCCS_QI2_AnimeOverhaulV2:0.8 —— **风格 LoRA 很关键**："
                         "不带它时 Qwen 会把网格渲的赛璐璐图重绘成半写实，"
                         "颜色变淡、锐度掉一档")
    ap.add_argument("--latent-lock-context",
                    help="顺序生成：ctx render 路径模板，支持 {key}/{i}。"
                         "与 --latent-lock-mask 同时给时启用 sampler 级 latent lock")
    ap.add_argument("--latent-lock-mask",
                    help="repair mask 路径模板（白=可去噪，黑=锁定），支持 {key}/{i}")
    ap.add_argument("--prompt", help="覆盖默认提示词")
    ap.add_argument("--compare", action="store_true", help="额外出对照拼图")
    args = ap.parse_args()

    size = args.size // 32 * 32
    # 只吃 JSON，不 import meow3d：本脚本跑在主项目 py3.13 venv 里
    import json
    plan = (json.loads(Path(args.plan).read_text(encoding="utf-8"))
            if args.plan else None)
    if not args.two_images:
        print("  单图模式：--ref 只作记录，**不参与生成**（只喂 beauty 底图）；"
              "原设计的颜色由后续 refine_views.py 的 flat 基准承载。\n"
              "  要把外部设计图强行注入用 --two-images（实测有「两格拼图」风险）",
              flush=True)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    gdir = Path(args.guide)
    ref = _open_rgba(args.ref)
    # 参考图统一铺到浅灰方画布（透明区当噪声会让 Qwen 出脏边）
    ref_sq = _fit(_flatten(ref), size)

    print(f"加载 Qwen（权重 {MODELS}）…", flush=True)
    t0 = time.time()
    eng = Engine(str(MODELS))
    pipe = eng.txt2img_pipe("qwen-image-2.1")
    print(f"  管线 {time.time() - t0:.1f}s", flush=True)

    cnu = None
    branch = None
    if args.mode in ("control", "both"):
        from workflows import controlnet_union as cnu_mod
        cnu = cnu_mod
        if not CN_PATH.is_file():
            print(f"缺 ControlNet 权重 {CN_PATH}")
            return 2
        t = time.time()
        branch = cnu.load_branch(str(CN_PATH), dtype=torch.bfloat16, device="cuda")
        print(f"  ControlNet 分支 {time.time() - t:.1f}s", flush=True)

    if not args.no_lightning:
        try:
            specs = [{"name": "qwen21_lightning", "scale": 1.0}]
            for spec in args.lora:
                name, _, sc = spec.partition(":")
                specs.append({"name": name, "scale": float(sc or 1.0)})
            eng.apply_loras(pipe, specs, scope="temp")
            args.steps = min(args.steps, 8)
            print(f"  已挂 LoRA {[s['name'] for s in specs]}（{args.steps} 步）",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  LoRA 挂载失败（不致命）: {e}", flush=True)
    elif args.lora:
        specs = []
        for spec in args.lora:
            name, _, sc = spec.partition(":")
            specs.append({"name": name, "scale": float(sc or 1.0)})
        eng.apply_loras(pipe, specs, scope="temp")
        print(f"  已挂 LoRA {[s['name'] for s in specs]}", flush=True)

    pipe.set_progress_bar_config(disable=True)
    lock = None
    if args.latent_lock_mask:
        # 顺序生成：sampler 级 latent lock（StableGen graph_validation.py:104）
        import qwen_latent_lock as _ll
        lock = _ll.LatentLock(pipe, size)
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    done = []
    for i in indices:
        key = f"v{i:02d}"
        g = gdir
        beauty = g / f"{key}_{args.guide_pass}.png"
        if not beauty.is_file():
            beauty = g / f"{key}_beauty.png"
        depth = g / f"{key}_depth_c.png"
        if args.mode in ("control", "both") and not depth.is_file():
            depth = g / f"{key}_depth.png"
        if args.mode in ("beauty", "both") and not beauty.is_file():
            print(f"缺底图 {beauty}，跳过")
            continue
        if args.mode in ("control", "both") and not depth.is_file():
            print(f"缺深度图 {depth}，跳过")
            continue

        az, el = 360.0 * i / args.cameras, 0.0
        if plan:
            for c in plan:
                if int(c.get("index", -1)) == i or c.get("name") == f"v{i:02d}":
                    az = float(c.get("azimuth", az))
                    el = float(c.get("elevation", 0.0))
                    break
        nearest = min(ANGLE_WORD,
                      key=lambda a: min(abs(az - a), 360.0 - abs(az - a)))
        gap = min(abs(az - nearest), 360.0 - abs(az - nearest))
        word = (ANGLE_WORD[nearest] if gap <= 22.5
                else f"{az:.0f} degree view")
        if el > 8:
            word += " from slightly above"
        elif el < -8:
            word += " from slightly below"
        kw = {}
        inj = None
        if args.mode in ("control", "both"):
            cond = Image.open(depth).convert("RGB")
            ctrl = cnu.encoder_condition(control=cond, vae=vae, size=(size, size),
                                         device="cuda", dtype=vae_dtype)
            inj = cnu.ControlInjection(pipe.transformer, branch, scale=args.scale)
            inj.set_condition(ctrl)
            kw["use_kv_cache"] = False    # 有 control 时必须关（见 controlnet_union 文档）

        # 条件图：默认只给彩图（--two-images 才把参考图一起塞进去）
        single = not bool(args.two_images)
        imgs = [] if single else [ref_sq]
        prompt = PROMPT_CONTROL.replace("an exact T-pose", f"an exact T-pose ({word})")
        if args.mode in ("beauty", "both"):
            imgs.append(_fit(_flatten(_open_rgba(beauty)), size))
            prompt = (PROMPT_EDIT_SINGLE if single else PROMPT_EDIT) \
                + f" The camera is {word}."
        if not imgs:
            imgs = [ref_sq]
            prompt = PROMPT_CONTROL
        kw["image"] = imgs if len(imgs) > 1 else imgs[0]
        if args.prompt:
            # --prompt 以前定义了却没用（死参数）。顺序生成要靠它传
            # 「保留非品红、只补品红」的 StableGen 原文提示词。
            prompt = args.prompt.replace("{word}", word)

        gen = torch.Generator("cuda").manual_seed(args.seed + i * 1000)
        if lock is not None:
            fmt = {"key": key, "i": i, "i02": f"{i:02d}"}
            mask_p = Path(args.latent_lock_mask.format(**fmt))
            if not mask_p.is_file():
                print(f"  [{key}] latent lock 缺 repair mask {mask_p}，跳过该图")
                continue
            lock.set_mask(mask_p)
            kw["callback_on_step_end"] = lock.callback
            print(f"  [{key}] latent lock grid={lock.info['grid']} "
                  f"repair={lock.info['repair']:.1%}", flush=True)

        try:
            t = time.time()
            out = pipe(prompt=prompt, width=size, height=size,
                       num_inference_steps=max(4, min(args.steps, 60)),
                       true_cfg_scale=1.0, negative_prompt=" ",
                       generator=gen,
                       output_resolution=size, **kw).images[0]
            dt = time.time() - t
        except torch.cuda.OutOfMemoryError:
            print(f"  [{key}] 显存不足 —— 换 --size 640 重试")
            return 3
        finally:
            if inj is not None:
                inj.remove()

        p = outdir / f"view_{i:02d}.png"
        out.convert("RGB").save(p)
        done.append((key, p))
        print(f"  [{key}] {word:24s} {dt:5.1f}s → {p.name}", flush=True)
        torch.cuda.empty_cache()

    if args.compare and done:
        tiles = []
        for key, p in done:
            tiles.append((key, Image.open(p).convert("RGB")))
        w, h = tiles[0][1].size
        sheet = Image.new("RGB", (w * len(tiles), h + 24), (24, 25, 28))
        d = ImageDraw.Draw(sheet)
        for i, (key, im) in enumerate(tiles):
            sheet.paste(im.resize((w, h)), (i * w, 24))
            d.text((i * w + 6, 6), key, fill=(230, 232, 236))
        cmp_path = outdir / "compare.png"
        sheet.save(cmp_path)
        print("对照图", cmp_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
