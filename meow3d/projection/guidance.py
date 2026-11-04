"""几何指引图（guidance map）：从目标机位渲出结构图，喂给 Qwen 生成对齐的新视角。

这是多相机投影从「不碎」走到「好看」的前提。之前 `view_02`（号称 90°）实际
还偏正面，投影到窄侧视 bbox 就被横向拉伸成脏斑 —— 根子不在投影，在**源视图
的几何本身不对**。给它一张同机位渲出来的深度图/素模，它就不用猜了。

与 StableGen 的 Guidance Map 是同一个思路（那边是 depth/normal workbench 渲染
喂 Image Edit），这里独立实现，不依赖其 GPL 代码。

产物与 `bake.projection_bbox()` **逐像素同框**：相机正交、视野 2.8、模型归一化
到 [-1,1]，所以渲染图里模型占的像素范围就是投影目标框，宿主侧把它交给 Qwen
后不需要二次对齐。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .bake import ROOT, ORTHO, blender_exe

PASSES = ("depth", "normal", "clay", "beauty")


def camera_list(count: int, indices: Sequence[int] | None = None,
                elevation: float = 0.0) -> List[Dict[str, Any]]:
    """按 `i*360/count` 生成相机；与 bake_projection 的 --cameras 约定一致。"""
    idx = list(indices) if indices else list(range(count))
    return [{"name": f"v{i:02d}", "azimuth": 360.0 * i / count,
             "elevation": elevation} for i in idx]


def render(mesh: str | Path, work_dir: str | Path,
           cameras: List[Dict[str, Any]], *, size: int = 1024,
           passes: Sequence[str] = PASSES,
           verbose: bool = True) -> Dict[str, Any]:
    """调 Blender 渲出每个机位的指引图，返回 {机位名: {pass: 路径}}。"""
    mesh = Path(mesh).resolve()
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    bad = [p for p in passes if p not in PASSES]
    if bad:
        raise ValueError(f"未知 pass {bad}，可用 {list(PASSES)}")
    cfg = {"mesh": str(mesh), "work_dir": str(work), "size": int(size),
           "ortho": ORTHO, "cameras": cameras, "passes": list(passes)}
    cfg_path = work / "guidance_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    if verbose:
        print(f"  渲指引图 {len(cameras)} 个机位 × {len(passes)} 个通道（{size}²）…",
              flush=True)
    subprocess.run([str(blender_exe()), "-b", "-P",
                    str(ROOT / "tools" / "blender_guidance.py"),
                    "--", str(cfg_path)], check=True)

    out: Dict[str, Dict[str, str]] = {}
    for c in cameras:
        d = {}
        for p in passes:
            f = work / f"{c['name']}_{p}.png"
            if f.is_file():
                d[p] = str(f)
        if d:
            out[c["name"]] = d
    if verbose:
        missing = [c["name"] for c in cameras if c["name"] not in out]
        print(f"  完成 {len(out)} 个机位" + (f"，缺 {missing}" if missing else ""),
              flush=True)
    return out


def depth_contrast(path: str | Path, lo: float = 0.15,
                   hi: float = 0.99) -> Image.Image:
    """把深度图的动态范围拉到满 [0,1]。

    Blender 的 MapRange 用的是固定的 [2.0,5.0] 深度区间，模型实际只占中间一段
    （归一化后身高 2.0，相机在 3.5 处 → 前后厚度通常只占 0.1~0.3），直接存 8 位
    会挤在很窄的灰阶里，ControlNet 分辨不出前后关系。这里按**非零像素**的
    分位数重映射，背景仍保持 0。
    """
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0
    fg = a > 0.02
    if not fg.any():
        return Image.open(path).convert("L")
    v = a[fg]
    vmin, vmax = float(np.percentile(v, lo * 100)), float(np.percentile(v, hi * 100))
    if vmax - vmin < 1e-6:
        return Image.open(path).convert("L")
    out = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    out[~fg] = 0.0
    return Image.fromarray((out * 255).astype(np.uint8), "L")


def prepare_depth(path: str | Path, dst: Optional[str | Path] = None) -> Path:
    """渲出来的深度图 → 可直接喂 ControlNet 的增强版（覆盖写或另存）。"""
    dst = Path(dst) if dst else Path(path).with_name(
        Path(path).stem.replace("_depth", "") + "_depth_c.png")
    depth_contrast(path).save(dst)
    return dst


def stack(views: Dict[str, Dict[str, str]], key: str,
          labels: Optional[Sequence[str]] = None) -> Optional[Image.Image]:
    """把多个机位的同一 pass 横向拼成一张（排查用）。"""
    names = labels or list(views.keys())
    ims = [Image.open(views[n][key]).convert("RGB") for n in names if n in views]
    if not ims:
        return None
    w, h = ims[0].size
    sheet = Image.new("RGB", (w * len(ims), h), (24, 25, 28))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h)), (i * w, 0))
    return sheet
