"""视图精修：把 Qwen 重绘从「整幅替换」降级为「受保护的局部修复」。

## 为什么需要这一层

StableGen 的 sequential 模式不是「每个机位独立出一张新贴图」，而是：

1. 先用**当前已投影好的纹理**从该机位渲一张 context render
   （`texturing/generator.py` 的 `export_visibility` / `export_emit_image`）；
2. 把未覆盖 / 低置信区域涂成品红，`QWEN_21_DEFAULT` 提示词明确要求
   ``Preserve its existing non-magenta textured regions``；
3. Qwen 的输出只是**补丁**，已贴好的像素一个都不动。

meow3d 之前把 Qwen 输出当成整张新贴图，于是 8 步 Lightning 的画风漂移
被全盘继承：橙发变浅金、正红战服变亮、赛璐璐边缘发软。更根本的是，
meow3d 的起点是**已经带贴图的 GLB**，StableGen 的起点是**白模**——
对白模来说 Qwen 是唯一的颜色来源，对带贴图网格来说整幅重绘是净损失。

## 本模块做什么（纯 numpy/cv2，不需要 ComfyUI / inpaint / 额外模型）

* `anchor_color` —— 在 YUV 空间把生成图前景的均值/方差对齐到同机位
  beauty 渲染，消除全局色调漂移，但保留生成图自己的明暗与边缘结构。
* `detail_gain` —— 比较生成图与 beauty 的局部高频能量：底图本来清晰
  （赛璐璐硬边）就不改，底图糊而生成图有细节（崩坏的脸）才放行。
* `protective_blend` —— 按 `protect` 控制「最多能改写多少像素」，
  `protect=0` 等价旧的整幅替换，`protect=1` 时最多只改写 35%，
  且只发生在生成图确实更清晰的区域。

视图与 beauty 必须**同机位同框**：先用 `bake.make_similarity_aligned_view`
把生成图对齐到 ProjUV 画布，guidance 的 beauty 与投影 bbox 逐像素同框，
因此两者可以直接逐像素合成，不会再引入一次对齐误差。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

Array = np.ndarray


def load_rgb(path) -> Array:
    """读成 float32 RGB [0,1]；直接取 RGB 通道，不乘 alpha。

    输入通常是已抠图的 RGBA：透明区的 RGB 往往是黑，若乘 alpha 会把半透明
    边缘一起压暗，颜色统计就会系统性偏暗。颜色锚定只在调用方给的 mask 内
    统计/生效，透明区本来就不该参与，所以这里不做合成。
    """
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def load_alpha(path, size: Optional[Tuple[int, int]] = None) -> Array:
    """读 alpha 为 float32 [0,1]；无 alpha 通道时返回全 1。"""
    im = Image.open(path)
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    if "A" not in im.getbands():
        return np.ones((im.height, im.width), np.float32)
    return np.asarray(im.getchannel("A"), dtype=np.float32) / 255.0


def background_mask(rgb: Array, tol: float = 0.10) -> Array:
    """用四角背景色估前景，返回 bool（True=前景）。

    guidance 的 beauty 是世界背景色纯色、不透明，四角必然是背景；
    Qwen 的浅灰背景不能靠颜色区分角色，所以这个函数只用于 beauty。
    """
    h, w = rgb.shape[:2]
    k = max(2, min(h, w) // 32)
    corners = np.concatenate([
        rgb[:k, :k].reshape(-1, 3), rgb[:k, -k:].reshape(-1, 3),
        rgb[-k:, :k].reshape(-1, 3), rgb[-k:, -k:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0)
    d = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    fg = d > tol
    # 去掉零散噪点，避免把抗锯齿边缘当背景
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((3, 3), np.uint8)).astype(bool)
    return fg


def has_color_reference(rgb: Array, mask: Array,
                        min_spread: float = 0.06) -> bool:
    """底图是否携带可信颜色信息。

    白模（没有 baseColorTexture）渲出的 flat/beauty 是**纯白或纯灰**：
    Emission fallback 灰 + Standard 变换 ≈ 0.98，通道间差只有 0.003。
    拿它做 YUV 颜色锚定会把 Qwen 刚生成的彩色全部拉成白，是灾难。
    带贴图 flat 的前景通道差通常 0.2~0.6，阈值 0.06 能干净分开。
    """
    m = mask.astype(bool)
    if m.sum() < 64:
        return False
    px = rgb[m]
    return float(np.mean(px.max(1) - px.min(1))) >= min_spread


def _yuv(rgb: Array) -> Array:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([
        0.299 * r + 0.587 * g + 0.114 * b,
        -0.14713 * r - 0.28886 * g + 0.436 * b,
        0.615 * r - 0.51499 * g - 0.10001 * b,
    ], axis=-1)


def _inv_yuv(yuv: Array) -> Array:
    y, u, v = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    return np.stack([
        y + 1.13983 * v,
        y - 0.39465 * u - 0.58060 * v,
        y + 2.03211 * u,
    ], axis=-1)


def anchor_color(src: Array, ref: Array, mask: Array,
                 strength: float = 1.0) -> Tuple[Array, dict]:
    """把 src 的颜色分布对齐到 ref，只统计并在 mask 内生效。

    YUV 三通道各做一次「均值对齐 + 受限的对比度对齐」。对比度系数被夹在
    [0.6, 1.6]，避免生成图本来就很平（或很爆）时被拉出伪色。mask 外原样
    保留，背景色不会污染统计、也不会被改成奇怪的颜色。

    返回 (结果图, 统计信息)，统计信息用于日志/回归测试。
    """
    if src.shape != ref.shape:
        raise ValueError(f"尺寸不一致: src{src.shape} ref{ref.shape}")
    if mask.shape != src.shape[:2]:
        raise ValueError("mask 尺寸不匹配")
    m = mask.astype(bool)
    if m.sum() < 64:
        return src.copy(), {"pixels": int(m.sum()), "applied": False}

    s, r = _yuv(src), _yuv(ref)
    out = s.copy()
    stats = {"pixels": int(m.sum()), "applied": True, "channels": []}
    for c in range(3):
        sv, rv = s[..., c][m], r[..., c][m]
        sm, rm = float(sv.mean()), float(rv.mean())
        ss, rs = float(sv.std()), float(rv.std())
        k = float(np.clip(rs / max(ss, 1e-3), 0.6, 1.6))
        k = 1.0 + (k - 1.0) * strength
        delta = (rm - sm) * strength
        out[..., c] = (s[..., c] - sm) * k + sm + delta
        stats["channels"].append({
            "src_mean": round(sm, 4), "ref_mean": round(rm, 4),
            "src_std": round(ss, 4), "ref_std": round(rs, 4),
            "k": round(k, 4), "delta": round(delta, 4),
        })
    matched = np.clip(_inv_yuv(out), 0.0, 1.0)
    result = src.copy()
    result[m] = matched[m]
    return result.astype(np.float32), stats


def _highfreq(rgb: Array, sigma: float = 1.6) -> Array:
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigma))


def detail_gain(src: Array, base: Array, mask: Array,
                sigma: float = 1.6) -> Array:
    """生成图比底图「多出多少细节」，返回 [0,1] 的逐像素放行权重。

    底图硬边清晰 / 平涂大色块 / 生成图更糊 → 0（不动）；
    底图糊成一片而生成图有五官、线条 → 接近 1（放行）。
    用前景内 90 分位归一化，避免整幅被一个高对比边缘拉爆。
    """
    ds, db = _highfreq(src, sigma), _highfreq(base, sigma)
    diff = ds - db
    m = mask.astype(bool)
    scale = float(np.percentile(np.abs(diff[m]), 90)) if m.any() else 1.0
    g = np.clip(diff / max(scale, 1e-3), 0.0, 1.0)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 3.0)
    g = np.clip(g, 0.0, 1.0)
    g[~m] = 0.0
    return g.astype(np.float32)


def protective_blend(src: Array, base: Array, mask: Array,
                     protect: float = 0.85, strength: float = 1.0,
                     sigma: float = 1.6) -> Tuple[Array, dict]:
    """颜色锚定 + 受保护合成。

    ``protect=0``   → 完全采用生成图（等价旧的整幅替换，但仍带颜色锚定）；
    ``protect=0.85``→ 基准只改写 15%，生成图更清晰处最多再加到 ~50%；
    ``protect=1``   → 基准不改写，只在生成图明显更清晰处最多改写 35%。

    白模路线（底图没有可信贴图）应该用 protect=0；
    从带贴图 GLB 出发（底图本来就对）应该用 protect>=0.8。
    """
    p = float(np.clip(protect, 0.0, 1.0))
    m = mask.astype(bool)
    anchored, stats = anchor_color(src, base, m, strength=strength)
    g = detail_gain(anchored, base, m, sigma=sigma)
    w = np.clip((1.0 - p) + p * 0.35 * g, 0.0, 1.0)
    out = base + (anchored - base) * w[..., None]
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    stats.update({
        "protect": p,
        "mean_weight": round(float(w[m].mean()), 4) if m.any() else 0.0,
        "max_weight": round(float(w.max()), 4),
        "mean_detail_gain": round(float(g[m].mean()), 4) if m.any() else 0.0,
    })
    return out, stats


def save_rgba(rgb: Array, alpha: Array, path) -> None:
    from pathlib import Path
    a = np.clip(alpha, 0.0, 1.0)
    arr = np.concatenate([
        (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        (a * 255).astype(np.uint8)[..., None],
    ], axis=2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(path)
