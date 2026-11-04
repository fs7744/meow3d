"""把普通图片抠成透明 PNG，供 3D 生成使用（**显式调用，不默认抠**）。

    python tools/matte.py 原图.png 透明图.png
    python tools/matte.py 原图.png 透明图.png --model u2net

模型缓存在 `~/.u2net/`（本机已有 u2net.onnx / birefnet-general-lite.onnx）。
需要 `onnxruntime`（pip install onnxruntime）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model", default="birefnet-general-lite")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        print("找不到", src)
        return 2
    dst.parent.mkdir(parents=True, exist_ok=True)

    t = time.time()
    im = Image.open(src).convert("RGB")
    out = remove(im, session=new_session(args.model))
    out.save(dst)
    a = np.array(out.convert("RGBA"))[:, :, 3]
    print(f"{dst}  {out.size}  透明 {((a < 20).mean()):.1%}  "
          f"不透明 {((a > 200).mean()):.1%}  {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
