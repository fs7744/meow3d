"""命令行：一张图 → 带原生纹理的 GLB（进程内，无需 ComfyUI 服务）。

    python tools/generate.py --plan                     # 只核对节点参数映射
    python tools/generate.py "D:/xxx/角色.png" --seed 42
    python tools/generate.py "D:/xxx/角色.png" --bg-removal auto  # 自动 BiRefNet 抠图
    python tools/generate.py "D:/xxx/角色.png" --stablegen-align  # 实验开关，已证伪，别用

⚠️ 2026-11-05：默认走官方 `MeshWithTexturing.json` 原值。曾尝试把 5 个参数
抄成 StableGen 默认值（4096 图集 / guidance 3.0 / 10 万面 / remove_inner_faces
false / 自动抠图），严格 A/B 证明会让模型退化成薄片，故默认关闭。详见 README。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import api, graph, runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="TRELLIS.2 图生 3D（进程内，无需 ComfyUI 服务）")
    ap.add_argument("image", nargs="?", help="输入图片路径")
    ap.add_argument("--workflow", default=str(api.DEFAULT_WORKFLOW))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", help="输出文件名前缀，默认 meow3d/<图片名>_<seed>")
    ap.add_argument("--texture-size", type=int)
    ap.add_argument("--target-faces", type=int)
    ap.add_argument("--max-size", type=int,
                    help="PreProcessImage 的最长边上限，默认 1024；2K 原图建议设 2048")
    ap.add_argument("--hq", action="store_true",
                    help="套用官方 HighQuality 纹理参数 + UV 图集 4096（更贴原图，但更慢）")
    ap.add_argument("--stablegen-align", action="store_true",
                    help="【实验，已证伪】把 5 个参数改成 StableGen 默认值；"
                         "严格 A/B 会让模型退化成薄片，默认关，不要启用")
    ap.add_argument("--bg-removal", choices=("auto", "skip"), default="skip",
                    help="skip=沿用输入自带 alpha（默认，旧行为）；"
                         "auto=BiRefNet 自动抠图（不透明原图时用）")
    ap.add_argument("--matte-model", default="birefnet-general-lite",
                    help="rembg 模型名，默认 birefnet-general-lite")
    ap.add_argument("--plan", action="store_true", help="只打印节点参数映射，不加载模型")
    args = ap.parse_args()

    align = args.stablegen_align

    if args.plan:
        wf = graph.load_workflow(args.workflow)
        kw = api.stablegen_align(wf, enabled=align)
        print(f"工作流: {args.workflow}  节点 {len(wf['nodes'])}  连线 {len(wf['links'])}")
        print("StableGen 参数对齐: 开（实验，已证伪）" if align
              else "StableGen 参数对齐: 关（官方示例原值，默认）")
        print(f"抠图: {args.bg_removal} / {args.matte_model}")
        for item in graph.plan(wf, kwargs_overrides=kw):
            if item.get("override"):
                print(f"[{item['id']}] {item['type']}  <外部注入>")
                continue
            print(f"[{item['id']}] {item['type']}")
            for k, v in item["kwargs"].items():
                s = json.dumps(v, ensure_ascii=False, default=str)
                print(f"    {k} = {s[:160]}")
        return 0

    if not args.image:
        ap.error("需要图片路径（或用 --plan 只看参数映射）")

    print("环境:", runtime.available())
    print("StableGen 参数对齐:", "开（实验，已证伪）" if align else "关",
          "| 抠图:", args.bg_removal, "/", args.matte_model)
    res = api.generate(
        args.image, workflow=args.workflow, seed=args.seed, prefix=args.prefix,
        texture_size=args.texture_size, target_faces=args.target_faces,
        max_size=args.max_size, high_quality=args.hq,
        align_stablegen=align, bg_removal=args.bg_removal,
        matte_model=args.matte_model,
    )
    print(f"\n完成，用时 {res['elapsed']}s")
    print("  带纹理:", res["textured"])
    print("  白模  :", res["white"])
    print("  全部  :", res["exports"])
    return 0 if res["textured"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
