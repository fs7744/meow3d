"""给已有 UV 的网格做投影贴图（不跑 TRELLIS.2 原生纹理链）。

    python tools/project_texture.py 模型.glb 原图.png --out out/xxx.glb
    python tools/project_texture.py 模型.glb 原图.png --out out/xxx.glb --reproject \
        --exponent 8        # 改混合参数，复用已缓存视图，不重新生成

商用注意：这条路径**不使用 nvdiffrast / nvdiffrec**（NVIDIA 非商用许可）。
输入网格若没有 UV，先用 tools/uv_unwrap.py 展开。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="带 UV 的 GLB")
    ap.add_argument("reference", help="原图（最好透明 PNG）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024, help="UV 图集边长")
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--exponent", type=float, default=3.0,
                    help="cos^k 视角权重指数，越大越倾向最正对的相机")
    ap.add_argument("--discard", type=float, default=90.0, help="丢弃掠射角（度）")
    ap.add_argument("--mode", choices=("replace", "blend", "qwen"), default="replace",
                    help="replace=正面原图直贴其余保留（默认，稳）；"
                         "qwen=完整 8 相机混合 + Qwen 生成的新视角；"
                         "blend=完整混合但只用几何重投影（实验性）")
    ap.add_argument("--views-dir", help="mode=qwen 时，tools/qwen_views.py 的输出目录")
    ap.add_argument("--fill", choices=("base", "inpaint"), default="base",
                    help="blend/qwen 模式下未覆盖 texel 的填法；"
                         "base=用原贴图打底，inpaint=从邻近颜色扩散（诊断用）")
    ap.add_argument("--no-dppg", action="store_true", help="关闭深度边界抑制")
    ap.add_argument("--dppg-margin", type=int, default=3)
    ap.add_argument("--cache-dir")
    ap.add_argument("--reproject", action="store_true",
                    help="复用缓存的相机视图与 UV 映射，只重算混合")
    args = ap.parse_args()

    res = run(args.mesh, args.reference, args.out,
              size=args.size, cameras=args.cameras, exponent=args.exponent,
              discard_deg=args.discard, dppg=not args.no_dppg,
              dppg_margin=args.dppg_margin, mode=args.mode,
              views_dir=args.views_dir, fill=args.fill,
              cache_dir=args.cache_dir, reuse_views=args.reproject)
    print(f"完成：{res['path']}  覆盖 {res['coverage']:.1%}  {res['elapsed']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
