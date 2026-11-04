"""把 Qwen 视图做成「受保护的局部修复」，输出可直接 --align frame 投影的 RGBA 视图。

典型用法（接续 tools/guidance.py 与 tools/qwen_guided_views.py）：

    .venv/Scripts/python.exe tools/refine_views.py <网格.glb> \
        --views .tmp/guided_xxx/ready --guide .tmp/guided_xxx/guide \
        --indices 0,1,2,3 --out .tmp/guided_xxx/refined --protect 0.85

输入 `--views` 是**已抠成 RGBA** 的 Qwen 出图（run_guided 第 3 步的产物），
`--guide` 是同机位的指引图。颜色基准**优先用 `*_flat.png`**（Emission 直出，
等于贴图原色），缺了才退回 `*_beauty.png`（带灯光，会二次打光，不推荐）。
输出与 ProjUV 逐像素同框，后续 bake 用 `--align frame`，不要再走 similarity。

这一步补的是 StableGen 的 visibility/context 语义：底图（同机位已贴纹理
渲染）是可信上下文，Qwen 输出只作为局部补丁，且颜色先锚定回底图。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import bake, refine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="与 guidance / bake 用的同一个 GLB")
    ap.add_argument("--views", required=True, help="已抠好的 RGBA 视图目录")
    ap.add_argument("--guide", required=True, help="guidance 目录（beauty/depth_c）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--indices", default="0,1,2,3")
    ap.add_argument("--size", type=int, default=0,
                    help="画布边长；0=跟随 beauty 尺寸")
    ap.add_argument("--protect", type=float, default=0.85,
                    help="保留底图强度 0~1；带贴图 GLB 建议 0.8~1，白模用 0")
    ap.add_argument("--anchor", type=float, default=1.0,
                    help="颜色锚定强度 0~1；0=不做（保留 Qwen 原色调）")
    ap.add_argument("--base", choices=("auto", "flat", "beauty"), default="auto",
                    help="颜色/纹理基准：flat=Emission 直出（推荐），beauty=带灯光渲染")
    ap.add_argument("--no-align", action="store_true",
                    help="输入视图已经同框（跳过 similarity 对齐）")
    args = ap.parse_args()

    mesh = Path(args.mesh).resolve()
    vdir, gdir, odir = Path(args.views), Path(args.guide), Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)
    if not mesh.is_file() or not vdir.is_dir() or not gdir.is_dir():
        print("网格 / --views / --guide 路径有问题")
        return 2
    idx = [int(x) for x in args.indices.split(",") if x.strip()]

    def base_path(i: int) -> tuple[Path, str]:
        if args.base == "flat":
            order = ("flat",)
        elif args.base == "beauty":
            order = ("beauty",)
        else:
            order = ("flat", "beauty")
        for kind in order:
            p = gdir / f"v{i:02d}_{kind}.png"
            if p.is_file():
                return p, kind
        return gdir / f"v{i:02d}_beauty.png", "missing"

    b0, kind0 = base_path(idx[0])
    if not b0.is_file():
        print(f"缺同机位颜色基准：{b0}（先跑 tools/guidance.py）")
        return 2
    size = args.size or Image.open(b0).width
    print(f"颜色基准：{kind0}（{'Emission 直出' if kind0 == 'flat' else '带灯光，建议改用 flat'}）",
          flush=True)

    V = bake.normalized_vertices(mesh)
    work = odir / ".aligned"
    work.mkdir(parents=True, exist_ok=True)
    report = {"mesh": str(mesh), "size": size, "protect": args.protect,
              "anchor": args.anchor, "views": []}
    for i in idx:
        src = vdir / f"view_{i:02d}.png"
        bp, kind = base_path(i)
        if not src.is_file() or not bp.is_file():
            print(f"  跳过 v{i:02d}：缺 {src.name if not src.is_file() else bp.name}")
            continue
        key = f"v{i:02d}"
        aligned = work / f"view_{i:02d}.png"
        info = {}
        if args.no_align:
            im = Image.open(src).convert("RGBA")
            if im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            im.save(aligned)
        else:
            az = 360.0 * i / args.cameras
            bbox = bake.projection_bbox(V, az, 0.0)
            ref_mask = None
            for cand in (f"{key}_depth_c.png", f"{key}_depth.png"):
                if (gdir / cand).is_file():
                    ref_mask = gdir / cand
                    break
            info = bake.make_similarity_aligned_view(
                src, aligned, bbox, size=size, ref_mask=ref_mask)

        beauty = refine.load_rgb(bp)
        if beauty.shape[0] != size:
            beauty = np.asarray(Image.fromarray(
                (np.clip(beauty, 0, 1) * 255).astype(np.uint8)
            ).resize((size, size), Image.LANCZOS), np.float32) / 255.0
        alpha = refine.load_alpha(aligned, (size, size))
        fg = refine.background_mask(beauty)
        mask = (alpha > 0.1) & fg
        src_rgb = refine.load_rgb(aligned)
        out, stats = refine.protective_blend(
            src_rgb, beauty, mask, protect=args.protect,
            strength=args.anchor)
        dst = odir / f"view_{i:02d}.png"
        refine.save_rgba(out, alpha * fg.astype(np.float32), dst)
        entry = {"view": key, "base": kind,
                 "out": str(dst), "coverage": round(float(mask.mean()), 4),
                 "align": info.get("align", "frame"),
                 "iou": info.get("iou"), **stats}
        report["views"].append(entry)
        print(f"  {key}: 前景 {mask.mean():.1%}，对齐 IoU {info.get('iou', '-')}，"
              f"平均改写权重 {stats['mean_weight']:.3f}，"
              f"色偏 ΔY {stats['channels'][0]['delta'] if stats.get('channels') else 0:+.3f}"
              f" → {dst.name}", flush=True)

    (odir / "refine_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {len(report['views'])} 张 → {odir}（画布 {size}²）")
    print("下一步：bake_projection.py --views-dir <out> --align frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
