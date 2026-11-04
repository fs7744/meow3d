"""渲指引图 CLI：从目标机位渲出深度/法线/素模/彩图，供 Qwen 生成对齐的新视角。

    # 4 个正交机位
    python tools/guidance.py out/meow3d/knight.glb --cameras 4 --indices 0,1,2,3

    # 只有 8 机位里的正/右/背/左
    python tools/guidance.py out/meow3d/knight.glb --cameras 8 --indices 0,2,4,6

产物落 `--out-dir`（默认 `.tmp/guide_<网格名>`），每个机位 6 个文件：

| 文件 | 用途 |
|---|---|
| `v00_depth.png` | ControlNet depth 条件（近白远黑，背景黑） |
| `v00_depth_c.png` | 同上，动态范围拉满（**实际喂给 ControlNet 用这张**） |
| `v00_normal.png` | 法线编码 RGB（排查几何用） |
| `v00_clay.png` | 素模（可做 lineart / 人眼检查） |
| `v00_beauty.png` | 带原贴图的彩图（`qwen_guided_views.py --mode beauty` 的底图） |
| `v00_flat.png` | **baseColor 直出（Emission，无光照）**：颜色锚定的可信基准 |

⚠️ `beauty` 带灯光高光，只能给 Qwen 当构图底图；**颜色基准必须用 `flat`**，
否则贴图会被写上「已经打过光的颜色」，渲染时二次打光 → 发灰发亮。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import guidance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="输入 GLB（白模也行，素模/深度不依赖贴图）")
    ap.add_argument("--out-dir", help="输出目录，默认 .tmp/guide_<网格名>")
    ap.add_argument("--cameras", type=int, default=4, help="总机位数（决定方位角步长）")
    ap.add_argument("--indices", default="", help="取哪些机位，如 0,2,4,6；留空=全部")
    ap.add_argument("--size", type=int, default=1024, help="输出边长（与 bake 同框）")
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--plan", help="相机方案 JSON（tools/plan_cameras.py 生成）；"
                                   "给了就用它的方位角/俯仰角，忽略 --cameras/--indices")
    ap.add_argument("--passes", default="depth,normal,clay,beauty,flat",
                    help="要渲哪些通道，逗号分隔")
    ap.add_argument("--sheet", action="store_true", help="额外拼 4 联对照图")
    args = ap.parse_args()

    mesh = Path(args.mesh)
    if not mesh.is_file():
        print("找不到网格", mesh)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else \
        ROOT / ".tmp" / f"guide_{mesh.stem}"
    idx = [int(x) for x in args.indices.split(",") if x.strip()] if args.indices else None
    if args.plan:
        from meow3d.projection import cameras
        plan = cameras.load_plan(args.plan)
        cams = [{"name": c["name"],
                 "azimuth": float(c.get("azimuth", 0)),
                 "elevation": float(c.get("elevation", 0))} for c in plan]
    else:
        cams = guidance.camera_list(args.cameras, idx, args.elevation)
    passes = tuple(p.strip() for p in args.passes.split(",") if p.strip())

    print(f"网格 {mesh}\n机位 {len(cams)} 个: "
          + ", ".join(f"{c['name']}@{c['azimuth']:.0f}°" for c in cams), flush=True)
    res = guidance.render(mesh, out_dir, cams, size=args.size, passes=passes)

    if "depth" in passes:
        for c in cams:
            if c["name"] in res and "depth" in res[c["name"]]:
                p = guidance.prepare_depth(res[c["name"]]["depth"])
                res[c["name"]]["depth_c"] = str(p)
        print("  深度图已做动态范围增强（_depth_c.png）", flush=True)

    if args.sheet:
        for key in ("depth_c", "clay", "beauty"):
            sheet = guidance.stack(res, key)
            if sheet is not None:
                p = out_dir / f"sheet_{key}.png"
                sheet.save(p)
                print("  对照图", p, flush=True)

    print(f"\n完成：{len(res)} 个机位 → {out_dir}")
    print("  下一步：tools/qwen_guided_views.py --guide "
          f"{out_dir} --mode control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
