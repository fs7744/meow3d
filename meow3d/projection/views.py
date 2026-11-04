"""每个相机的「视图」怎么来。

StableGen 用 SDXL / Qwen Image Edit 逐相机**生成**视图，并把已贴好的纹理渲染图
作为下一视角的上下文。这里把这一步抽象成可插拔的 ViewProvider：

- `HybridViewProvider`：正面相机直接搬**原图像素**（脸部高清直贴），
  其余相机把「当前累积贴图」按几何重投影成该视角图（sequential context 的
  等价实现）。可选传入一份已有 UV 贴图当底图，保留侧背细节。
- `DiffusionViewProvider`：接口桩，接 Qwen-Image-Edit / SDXL 时在这里实现，
  其余模块完全不用改。

全部自己实现，不复制 StableGen 的 GPL 源码。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from PIL import Image

from .geometry import Camera, UVMap


@dataclass
class View:
    image: np.ndarray        # (S,S,3) float32 0~1
    mask: np.ndarray         # (S,S) float32 0/1，哪些像素可用


def _fit_reference(image: Image.Image, target: int, screen_bbox: tuple) -> tuple[np.ndarray, np.ndarray]:
    """把原图主体 bbox 裁成方形，缩放到正面相机的屏幕 bbox，返回 (图, mask)。"""
    rgba = image.convert("RGBA")
    a = np.asarray(rgba)[:, :, 3]
    ys, xs = np.where(a > 20)
    if len(xs) == 0:
        box = (0, 0, rgba.width, rgba.height)
    else:
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crop = rgba.crop(box)
    side = max(crop.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))

    x0, y0, x1, y1 = screen_bbox
    w = max(8, int(x1 - x0))
    h = max(8, int(y1 - y0))
    sq = square.resize((w, h), Image.LANCZOS)
    arr = np.asarray(sq).astype(np.float32) / 255.0

    canvas = np.zeros((target, target, 3), np.float32)
    mask = np.zeros((target, target), np.float32)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(target, x0 + w), min(target, y0 + h)
    if xb <= xa or yb <= ya:
        return canvas, mask
    sub = arr[ya - y0: yb - y0, xa - x0: xb - x0]
    alpha = sub[..., 3:4]
    canvas[ya:yb, xa:xb] = sub[..., :3] * alpha
    mask[ya:yb, xa:xb] = alpha[..., 0]
    return canvas, mask


def scatter_atlas_to_view(uvm: UVMap, cam: Camera, atlas: np.ndarray,
                          size: int) -> View:
    """把 UV 图集按几何重投影成该相机看到的图（z-buffer 取最近 texel）。"""
    scr = cam.screen(uvm.points)
    dep = cam.depth(uvm.points)
    facing = np.dot(uvm.normals, cam.direction) > 0
    ok = uvm.valid & facing
    px = np.clip((scr[..., 0] * size).astype(np.int32), 0, size - 1)
    py = np.clip((scr[..., 1] * size).astype(np.int32), 0, size - 1)
    flat_idx = (py * size + px)[ok]
    flat_dep = dep[ok]
    flat_col = atlas[ok]

    image = np.zeros((size, size, 3), np.float32)
    mask = np.zeros((size, size), np.float32)
    if flat_idx.size == 0:
        return View(image, mask)

    order = np.lexsort((flat_dep, flat_idx))
    idx_s = flat_idx[order]
    last = np.r_[idx_s[1:] != idx_s[:-1], True]
    chosen = idx_s[last]
    col = flat_col[order][last]
    image.reshape(-1, 3)[chosen] = col
    mask.reshape(-1)[chosen] = 1.0
    return View(image, mask)


class ViewProvider:
    def build(self, uvm: UVMap, cams: List[Camera], atlas: np.ndarray,
              size: int) -> List[Optional[View]]:
        raise NotImplementedError


class HybridViewProvider(ViewProvider):
    """正面 = 原图像素；其余 = 当前累积贴图的重投影（可选底图打底）。"""

    def __init__(self, reference: str | Path, base_atlas: Optional[np.ndarray] = None,
                 front_index: int = 0):
        self.reference = Image.open(reference)
        self.base_atlas = base_atlas
        self.front_index = front_index

    def build(self, uvm: UVMap, cams: List[Camera], atlas: np.ndarray,
              size: int) -> List[Optional[View]]:
        # 正面视图的有效屏幕范围：由朝向正面相机的 texel 决定
        cam0 = cams[self.front_index]
        scr0 = cam0.screen(uvm.points)
        ok = uvm.valid & (np.dot(uvm.normals, cam0.direction) > 0)
        if ok.any():
            xs, ys = scr0[..., 0][ok], scr0[..., 1][ok]
            bbox = (int(xs.min() * size), int(ys.min() * size),
                    int(xs.max() * size), int(ys.max() * size))
        else:
            bbox = (int(size * 0.1), int(size * 0.1), int(size * 0.9), int(size * 0.9))

        source = atlas if self.base_atlas is None else self.base_atlas
        out: List[Optional[View]] = []
        for i, cam in enumerate(cams):
            if i == self.front_index:
                img, msk = _fit_reference(self.reference, size, bbox)
                # 底图打底：原图透明处必须换成底图颜色，不能留黑；
                # 同时这些区域仍然算「有视图」，否则会被误判成空洞。
                base = scatter_atlas_to_view(uvm, cam, source, size)
                combined = img * msk[..., None] + base.image * (1 - msk[..., None])
                combined_mask = np.maximum(msk, base.mask)
                out.append(View(combined, combined_mask))
            else:
                out.append(scatter_atlas_to_view(uvm, cam, source, size))
        return out


class QwenViewProvider(ViewProvider):
    """消费 `tools/qwen_views.py` 用 Qwen-Image-2.1 生成的多视角图。

    每个文件 `view_XX.png` 对应第 XX 个相机（`make_cameras` 的方位角顺序一致）。
    视图是全画面人像，坐标与 `Camera.screen()` 的 [0,1] 屏幕空间对齐。
    这样每个相机都有**新信息**（不再是 UV 图集自身的重投影），
    完整多相机混合才有意义 —— 这正是 StableGen 的核心。
    """

    def __init__(self, view_dir: str | Path):
        self.view_dir = Path(view_dir)

    def build(self, uvm: UVMap, cams: List[Camera], atlas: np.ndarray,
              size: int) -> List[Optional[View]]:
        out: List[Optional[View]] = []
        for i, cam in enumerate(cams):
            p = self.view_dir / f"view_{i:02d}.png"
            if not p.is_file():
                print(f"  [QwenViewProvider] 缺 {p.name}，该相机跳过", flush=True)
                out.append(None)
                continue
            im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            # 浅灰背景抑制：Qwen 出图背景约 (200,202,206)，银甲偏白/带蓝影，
            # 用宽松阈值只压掉明确的背景，避免把铠甲一起剔掉。
            bg = np.array([0.784, 0.792, 0.808], np.float32)
            dist = np.linalg.norm(arr - bg, axis=-1)
            mask = np.clip((dist - 0.035) / 0.075, 0.15, 1.0).astype(np.float32)
            out.append(View(arr, mask))
        return out


class DiffusionViewProvider(ViewProvider):
    """接任意 2D 生成模型的通用接口。

    StableGen 的做法是：渲染该相机的 Depth/Normal 图当结构参考，
    把已贴好的纹理渲染图当上下文，让模型生成/编辑这一视角。
    本模块其余部分（权重、DPPG、混合、Reproject）无需改动。

    当前已落地的实现见 `QwenViewProvider`。
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "通用 DiffusionViewProvider 尚未实现；"
            "用 QwenViewProvider（tools/qwen_views.py 产出）或 HybridViewProvider")
