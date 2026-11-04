"""v2 投影贴图 CLI：Blender 3D 表面投影 + bake（不调用 NVIDIA 库）。

单相机最小闭环（正面原图像素直贴，验证 bake 不碎）：
    python tools/bake_projection.py <网格.glb> --reference <透明角色.png> \
        --out out/meow3d/knight_baked.glb

多相机（Qwen 生成的新视角，方位角按 i*360/N）：
    python tools/bake_projection.py <网格.glb> --views-dir .tmp/qwen_views \
        --cameras 4 --view-indices 0,2,4,6 --out out/meow3d/knight_baked4.glb

推荐：先用 tools/guidance.py 渲同机位结构图，再让 Qwen 按它重绘，
然后用 --align similarity 对齐（等比缩放+平移，别用 bbox 分轴拉伸）：
    python tools/guidance.py 网格.glb --cameras 4 --indices 0,1,2,3
    python tools/qwen_guided_views.py --guide .tmp/guide_xxx --mode beauty ...
    python tools/bake_projection.py 网格.glb --reference 原图.png \
        --views-dir .tmp/qwen_guided_matte --cameras 4 --view-indices 1,2,3 \
        --align similarity --guide-dir .tmp/guide_xxx --out out/xxx.glb

自定义：
    --view front:path/to.png:0 --view back:path/to.png:180
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import bake as bake_impl  # noqa: E402
from meow3d.projection import cameras  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="输入 GLB（白模或带贴图均可；会在 Blender 里重新 unwrap）")
    ap.add_argument("--out", required=True, help="输出 GLB")
    ap.add_argument("--reference", help="单相机模式：正面原图（最好带 alpha）")
    ap.add_argument("--views-dir", help="多相机模式：tools/qwen_views.py 的输出目录")
    ap.add_argument("--cameras", type=int, default=8, help="总相机数（决定方位角步长）")
    ap.add_argument("--view-indices", default="",
                    help="只取哪些视角，如 0,2,4,6；留空取前 N 个")
    ap.add_argument("--plan", help="相机方案 JSON；给了就用它的方位角/俯仰角")
    ap.add_argument("--view", action="append", default=[],
                    help="自定义视角 name:图片路径:方位角，可重复")
    ap.add_argument("--size", type=int, default=0,
                    help="UV 图集边长；0=自动（keep 模式跟原生贴图同边长，否则 1024）")
    ap.add_argument("--exponent", type=float, default=3.0, help="cos^k 权重指数")
    ap.add_argument("--margin", type=int, default=4, help="bake 边缘扩张 px")
    ap.add_argument("--decimate", type=int, default=0, help="降面目标（0=不降）")
    ap.add_argument("--engine", default="CYCLES",
                    choices=("BLENDER_EEVEE_NEXT", "CYCLES"),
                    help="bake 必须用 CYCLES（EEVEE Next 不支持 object.bake）")
    ap.add_argument("--unwrap", default="keep",
                    choices=("keep", "cube", "smart", "auto"),
                    help="keep=保留原始不重叠 UV（默认，最干净）；cube/smart=重新展开")
    ap.add_argument("--fill", default="base", choices=("base", "diffuse"),
                    help="base=未覆盖 texel 用原生贴图兜底；diffuse=邻域扩散")
    ap.add_argument("--blend", default="weighted", choices=("weighted", "winner"),
                    help="winner=每个 texel 只取权重最大的相机（无 raycast 时更干净）")
    ap.add_argument("--detail", type=float, default=0.0,
                    help="借原生贴图高频的强度 0~1（投影会放大原图 → 糊，"
                         "0.5 左右补回金属/布料细节；需要 --fill base")
    ap.add_argument("--no-occlude", action="store_true",
                    help="关掉遮挡剔除（默认开；关掉后侧视会把被挡住的背面颜色投到正面）")
    ap.add_argument("--align", default="bbox",
                    choices=("bbox", "similarity", "frame"),
                    help="bbox=按 alpha 包围盒 x/y 各自拉满（旧行为）；"
                         "similarity=等比缩放+平移对齐到模型框（Qwen 引导生成的图用这个）；"
                         "frame=已与 ProjUV 同框，原样铺满")
    ap.add_argument("--guide-dir",
                    help="--align similarity 的轮廓基准（tools/guidance.py 的输出目录）")
    ap.add_argument("--work-dir", help="缓存目录（含对齐视图与 bake 结果）")
    ap.add_argument("--reproject", action="store_true",
                    help="复用已烘焙的 npy，只重算合并与导出")
    args = ap.parse_args()

    views = []
    align = args.align
    plan = cameras.load_plan(args.plan) if args.plan else None
    guide_dir = Path(args.guide_dir) if args.guide_dir else None
    if args.reference:
        views.append({"name": "front", "image": args.reference,
                      "azimuth": 0.0, "elevation": 0.0, "align": "bbox"})
    if args.views_dir:
        vd = Path(args.views_dir)
        if args.view_indices:
            idx = [int(x) for x in args.view_indices.split(",") if x.strip()]
        else:
            idx = list(range(args.cameras))
        for i in idx:
            p = vd / f"view_{i:02d}.png"
            if not p.is_file():
                print(f"跳过缺失视角 {p}")
                continue
            az, el = (cameras.angles_for(plan, i) if plan
                      else (360.0 * i / args.cameras, 0.0))
            v = {"name": f"v{i:02d}", "image": str(p),
                 "azimuth": az, "elevation": el,
                 "align": align}
            if align == "similarity" and guide_dir is not None:
                # 指引图的深度通道 = 模型在该机位的真实轮廓，做 IoU 基准
                for cand in (f"v{i:02d}_depth_c.png", f"v{i:02d}_depth.png"):
                    if (guide_dir / cand).is_file():
                        v["ref_mask"] = str(guide_dir / cand)
                        break
            views.append(v)
    for spec in args.view:
        parts = spec.split(":")
        if len(parts) < 2:
            print("格式应为 name:路径[:方位角]", spec)
            return 2
        views.append({"name": parts[0], "image": parts[1],
                      "azimuth": float(parts[2]) if len(parts) > 2 else 0.0,
                      "elevation": 0.0})
    if not views:
        print("至少给一个视图：--reference / --views-dir / --view")
        return 2

    for v in views:
        if not Path(v["image"]).is_file():
            print("找不到视图图", v["image"])
            return 2

    print(f"网格 {args.mesh}\n视图 {len(views)} 个: "
          + ", ".join(f"{v['name']}@{v['azimuth']:.0f}°" for v in views), flush=True)
    res = bake_impl.run(
        args.mesh, args.out, views, size=args.size, exponent=args.exponent,
        margin=args.margin, decimate=args.decimate, engine=args.engine,
        unwrap=args.unwrap, fill=args.fill, blend=args.blend, detail=args.detail,
        occlude=not args.no_occlude,
        work_dir=args.work_dir, reproject=args.reproject,
    )
    print(f"\n完成：{res['path']}（{res.get('out_size', 0) / 1e6:.1f}MB）")
    print(f"  视图 {res['views']} 个，覆盖 {res['coverage']:.1%}，"
          f"导出方式 {res.get('export', '?')}，缓存 {res['work_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
