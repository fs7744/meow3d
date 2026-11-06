"""一张图 → 带 PBR 贴图的 GLB，跑 StableGen 的 `prompt_text_trellis2` 9 节点链。

    python tools/generate_sg.py --plan
    python tools/generate_sg.py "out/in/knight_matte.png" --seed 42
    python tools/generate_sg.py "out/in/knight_tpose.png" --bg-removal auto

与 tools/generate.py 的区别：那个跑 visualbruno 的 24 节点 UI 工作流，
这个跑 StableGen 指定的 PozzettiAndrea/ComfyUI-TRELLIS2（pin 9b878516）
的 9 节点 **API 格式 prompt**，节点实现与 StableGen 完全一致，进程内执行，
不启动 ComfyUI 服务。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import sg_prompt, sg_runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="StableGen prompt_text_trellis2 9 节点链（进程内）")
    ap.add_argument("image", nargs="?", help="输入图片，建议已抠成 RGBA")
    ap.add_argument("--seed", type=int, default=0, help="对应 trellis2_seed，默认 0")
    ap.add_argument("--prefix", default="trellis2")
    ap.add_argument("--bg-removal", choices=("skip", "auto"), default="skip",
                    help="skip=StableGen 的 skip 分支（用输入自带 alpha，默认）；"
                         "auto=跑 Trellis2RemoveBackground（需 BiRefNet 权重）")
    ap.add_argument("--texture-size", type=int, default=4096)
    ap.add_argument("--target-faces", type=int, default=100000)
    ap.add_argument("--tex-guidance", type=float, default=3.0)
    ap.add_argument("--background-color", choices=("black", "gray", "white"),
                    default="black",
                    help="Trellis2GetConditioning 原生选项；输入是白底原图时试 white")
    ap.add_argument("--shape-only", action="store_true",
                    help="对应 trellis2_skip_texture=True：走 prompt_text_trellis2_shape_only "
                         "出白模（StableGen 选 qwen_image_21 贴图模式时的 mesh 阶段）")
    ap.add_argument("--plan", action="store_true", help="只打印最终 prompt，不加载模型")
    args = ap.parse_args()

    image = args.image or "out/in/knight_matte.png"
    prompt = sg_prompt.build_prompt(
        image=image, seed=args.seed, prefix=args.prefix,
        bg_removal=args.bg_removal, texture_size=args.texture_size,
        target_face_count=args.target_faces,
        tex_guidance_strength=args.tex_guidance,
        background_color=args.background_color,
        skip_texture=args.shape_only,
    )

    if args.plan:
        print(f"输入图: {image} | 背景去除: {args.bg_removal} | "
              f"节点 {len(prompt)} | 白模 {args.shape_only}")
        for nid in sg_prompt._topo(prompt):
            node = prompt[nid]
            print(f"[{nid}] {node['class_type']}")
            for k, v in node["inputs"].items():
                s = json.dumps(v, ensure_ascii=False)
                print(f"    {k} = {s[:150]}")
        return 0

    if not Path(image).is_file():
        ap.error(f"找不到输入图：{image}")
    st = sg_runtime.available()
    if not st["ok"]:
        ap.error("环境未就绪：" + "；".join(st["reasons"]))

    def log(nid, ctype, phase):
        if phase == "start":
            print(f"  → [{nid}] {ctype}", flush=True)

    print("环境:", st)
    res = sg_prompt.execute(prompt, image=image, on_node=log)
    print(f"\n完成，用时 {res['elapsed']}s")
    for p in res["exports"]:
        print("  导出:", p)
    return 0 if res["exports"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
