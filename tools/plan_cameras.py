"""生成相机布置方案（normal-weighted / 等分环）。

    .venv/Scripts/python.exe tools/plan_cameras.py <网格.glb> --count 8 \
        --out .tmp/cameras8.json
    .venv/Scripts/python.exe tools/plan_cameras.py <网格.glb> --count 4 \
        --ring --out .tmp/ring4.json      # 旧行为对照

产物 JSON 可直接喂给 guidance.py / qwen_guided_views.py / refine_views.py /
bake_projection.py / run_guided.py 的 --plan / --camera-plan，保证五个环节
用的是**同一套相机角度**（这是多视图投影最容易出错的地方）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import cameras  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--max-elevation", type=float, default=30.0)
    ap.add_argument("--exclude-bottom", type=float, default=60.0)
    ap.add_argument("--ring", action="store_true",
                    help="用旧的等分水平环（对照用）")
    ap.add_argument("--ring-elevation", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mesh = Path(args.mesh)
    if not mesh.is_file():
        print("找不到网格", mesh)
        return 2
    if args.ring:
        plan = cameras.ring_plan(args.count, elevation=args.ring_elevation)
    else:
        plan = cameras.normal_weighted_plan(
            mesh, count=args.count, max_elevation=args.max_elevation,
            exclude_bottom_deg=args.exclude_bottom, seed=args.seed)
    p = cameras.save_plan(plan, args.out)
    print(f"{len(plan)} 个机位 → {p}")
    for c in plan:
        print(f"  {c['name']}  az={c['azimuth']:7.2f}°  el={c['elevation']:6.2f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
