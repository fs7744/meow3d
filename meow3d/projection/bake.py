"""v2 投影后端：投影发生在 3D 表面，UV 贴图只是 bake 的产物。

按 StableGen Wiki 公开的**架构思路**独立实现，未复制其 GPL-3.0 源码，
不调用 nvdiffrast / nvdiffrec：

  1. 宿主侧：把每张视图按模型在该相机下的投影 bbox 严格对齐（RGBA 方形图）。
  2. Blender：重新 unwrap（Cube Project）得到干净的 BakeUV；每个相机生成一份
     ProjUV（相机矩阵算出的投影坐标）；用 Emission 材质把「视图颜色」和
     「alpha * cos^k 权重」分别 bake 到 BakeUV。
     bake 是逐三角面正向光栅化 + 重心插值，因此不会出现 texel 采样跳变。
  3. 宿主侧：numpy 加权合并所有相机的 color/weight，补洞，写 atlas PNG。
  4. 出口两条路：
     * unwrap=keep（UV 没变）→ **二进制替换原 GLB 的 baseColorTexture**，
       网格一字节不动（trimesh 重导出会丢 metallicRoughness 贴图，
       Blender 往返会把 36 万顶点拆成 135 万、21MB 变 50MB）。
     * 重新 unwrap → Blender 贴 atlas 导出新网格。

视图 bake 结果缓存在 work_dir/views/*.npy；只改混合参数时可用 --reproject
跳过 Blender 重烘（Reproject）。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import trimesh
from PIL import Image

from . import glbpatch

ROOT = Path(__file__).resolve().parents[2]
BLENDER_CANDIDATES = [
    Path(r"D:\code\gen\tools\blender\blender-4.5.9-windows-x64\blender.exe"),
    Path(r"D:\GAMES_tool\blender\blender.exe"),
]

ORTHO = 2.8          # 正交相机视野边长；与 blender_bake_views.py 必须一致
NORM_SCALE = 2.0     # 归一化到 [-1,1]；与 blender_preview.py 一致


def blender_exe() -> Path:
    for p in BLENDER_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("找不到 Blender，改 bake.BLENDER_CANDIDATES")


def normalized_vertices(mesh_path: str | Path) -> np.ndarray:
    """读 GLB（应用 scene graph 变换），按 Blender 侧相同的公式归一化。"""
    s = trimesh.load(str(mesh_path), force="scene")
    chunks = []
    for node in s.graph.nodes_geometry:
        T, geo_name = s.graph.get(node)
        g = s.geometry[geo_name]
        chunks.append(trimesh.transform_points(np.asarray(g.vertices, np.float64), T))
    if not chunks:
        raise ValueError("GLB 里没有几何体")
    V = np.vstack(chunks)
    # glTF 是 Y-up，Blender 导入时绕 X 轴 +90° 转成 Z-up：(x,y,z) -> (x,-z,y)。
    # trimesh 读的是 glTF 原始坐标，不转换的话算出来的「上」会是前后方向，
    # 视图 bbox 会扁平（实测 v 范围只有 0.19，模型高度被当成深度）。
    V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)
    lo, hi = V.min(0), V.max(0)
    center = (lo + hi) / 2
    scale = NORM_SCALE / float(np.max(hi - lo))
    return (V - center) * scale


def projection_bbox(V: np.ndarray, azimuth_deg: float,
                    elevation_deg: float = 0.0,
                    ortho: float = 0.0) -> tuple[float, float, float, float]:
    """模型在该相机下占的 [0,1]² 屏幕范围（u0,v0,u1,v1，v 向上）。

    ``ortho=0`` 用全局 ORTHO；传值则覆盖（裁满画幅时用）。
    这个值必须与 Blender 侧 ``cam_data.ortho_scale`` **完全一致**，
    否则 guidance 图、Qwen 出图、ProjUV 三者会错位。
    """
    scale = float(ortho) if ortho else ORTHO
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    d = np.array([math.sin(az) * math.cos(el),
                  -math.cos(az) * math.cos(el), math.sin(el)])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, d)
    right /= np.linalg.norm(right) + 1e-8
    u = 0.5 + (V @ right) / scale
    v = 0.5 + (V @ up) / scale
    return float(u.min()), float(v.min()), float(u.max()), float(v.max())


def fit_ortho(V: np.ndarray, angles: Sequence[tuple[float, float]],
              margin: float = 1.08, base: float = 0.0) -> float:
    """按所有相机方向下模型的最大投影跨度，算出「刚好装得下」的 ortho。

    原来是固定 ``ORTHO=2.8``，而模型归一化后最大跨度只有 2.0 —— 角色只占
    ~71% 画幅，1024² 视图里头部约 90px，投进 2048² 图集要放大 6~8 倍。
    把所有环节一起换成 ``span*margin``（默认 1.08，留 8% 边距），
    头部有效像素可提升约 1.3 倍，且 guidance/Qwen/ProjUV 仍然逐像素同框。

    ``margin`` 不能取 1.0：扩散模型对贴边内容容易裁掉头发/脚尖。
    """
    b = float(base) if base else ORTHO
    spans = []
    for az, el in angles:
        u0, v0, u1, v1 = projection_bbox(V, az, el, ortho=b)
        spans.append(max(u1 - u0, v1 - v0) * b)
    if not spans:
        return b
    return round(max(spans) * max(1.01, float(margin)), 4)


def make_aligned_view(src: str | Path, dst: str | Path, bbox: Sequence[float],
                      size: int = 1024) -> Dict[str, Any]:
    """把视图主体 bbox 映射到模型投影 bbox，输出 RGBA 方形 PNG。

    透明像素 alpha=0，Blender 侧会把它排除在权重之外，避免背景色混进贴图。
    """
    im = Image.open(src).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    transparent = float((a < 20).mean())
    if transparent > 0.05:
        ys, xs = np.where(a > 20)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        box = (0, 0, im.width, im.height)
    crop = im.crop(box)

    u0, v0, u1, v1 = bbox
    x0, x1 = int(round(u0 * size)), int(round(u1 * size))
    y0, y1 = int(round((1 - v1) * size)), int(round((1 - v0) * size))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(size, x1), min(size, y1)
    w, h = max(8, x1 - x0), max(8, y1 - y0)
    fitted = crop.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(fitted, (x0, y0))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return {"src": str(src), "transparent": round(transparent, 3),
            "source_box": list(box), "target_box": [x0, y0, x1, y1]}


def _mask_of(im: Image.Image, thr: int = 128) -> np.ndarray:
    a = np.asarray(im.convert("RGBA"))[:, :, 3]
    return a > thr


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _split_side_by_side(mask: np.ndarray,
                        ref_aspect: Optional[float]) -> Optional[np.ndarray]:
    """并排双主体 → 只留目标那一个；否则返回 None（不动）。

    Qwen 拿到「参考图 + 彩图」两张输入时，会时不时把结果画成**左右两格**
    （左边原图、右边结果），尤其是两个角色都是全身站姿时。这样的图直接投影会把
    两个人都糊到模型上。

    判定条件刻意收得紧，避免把「角色 + 分离的披风/武器」误切成两半：
      · 至少两个连通块各占 25% 以上面积；
      · 它们的 **x 区间互不相交**（真正的并排，而不是前后叠着）；
    取哪一个：宽高比更接近模型轮廓的那个（侧视很窄、正面很宽，这个判据够用）。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    total = float(mask.sum())
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.25 * total]
    if len(comps) < 2:
        return None
    spans = sorted((int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]))
                   for i in comps)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if b0 < a1:                     # x 有重叠 → 不是并排
            return None
    if ref_aspect is None:
        pick = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    else:
        def _ratio(i):
            w = float(stats[i, cv2.CC_STAT_WIDTH])
            h = max(1.0, float(stats[i, cv2.CC_STAT_HEIGHT]))
            return abs(math.log((w / h) / max(1e-6, ref_aspect)))
        pick = min(comps, key=_ratio)
    return lab == pick


def make_similarity_aligned_view(src: str | Path, dst: str | Path,
                                 bbox: Sequence[float], size: int = 1024,
                                 ref_mask: Optional[str | Path] = None,
                                 refine: bool = True,
                                 scale_mode: str = "area",
                                 core_iou: bool = False) -> Dict[str, Any]:
    """把生成图的主体做**相似变换**（等比缩放 + 平移，不改长宽比）对齐到模型投影框。

    为什么不能像 `make_aligned_view` 那样 x/y 各自拉满：Qwen 出图的宽高比本来就
    接近正确（实测侧视 0.243 vs 指引 0.246），强行分轴缩放只会把本来对的形状
    捏坏；而它的**大小和位置是漂的**（实测高占比 0.855 vs 0.703、中心偏左 0.1），
    所以真正需要修的是尺度和平移。

    尺度初值：有指引轮廓时用**面积比**（sqrt），没有才退回包围盒高度比。
    指引图的轮廓当基准做两级 IoU 搜索，把平移和残余尺度一起定下来。

    **所有面积/长度都先在 size 画布上量再相除** —— 源图通常是 768²、画布 2048²，
    直接拿两边像素数算比值会差 (size/src)² 倍（实测侧视被缩成 0.18 倍，投影直接
    空掉）。这个单位错误很容易写出来，也很难一眼看出来。
    """
    im = Image.open(src).convert("RGBA")
    mask = _mask_of(im)
    if not mask.any():
        raise ValueError(f"{src} 没有不透明像素（先跑 tools/matte.py 抠图）")

    u0, v0, u1, v1 = bbox
    tx0, tx1 = u0 * size, u1 * size
    ty0, ty1 = (1 - v1) * size, (1 - v0) * size
    th = max(8.0, ty1 - ty0)

    ref = None
    if ref_mask is not None and Path(ref_mask).is_file():
        rm = np.asarray(Image.open(ref_mask).convert("L").resize((size, size),
                                                                 Image.NEAREST))
        # 背景是**精确 0**（Blender 深度图无交点 = 0），所以阈值取 1 就够；
        # >127 会把「远的一半身体」切掉，面积比直接偏小一半。
        ref = rm > 1
    ref_aspect = None
    if ref is not None and ref.any():
        rx0, ry0, rx1, ry1 = _bbox_of(ref)
        ref_aspect = (rx1 - rx0) / max(1.0, ry1 - ry0)

    # 并排双主体（Qwen 的「前后对比」式出图）先只留目标那一个
    kept = _split_side_by_side(mask, ref_aspect)
    split = kept is not None
    if split:
        rgba = np.asarray(im).copy()
        rgba[~kept] = 0
        im = Image.fromarray(rgba, "RGBA")
        mask = kept

    sx0, sy0, sx1, sy1 = _bbox_of(mask)

    # 把源图掩膜量到 size 尺度上再比（见 docstring 的单位说明）
    m_size = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((size, size), Image.NEAREST)) > 127
    if m_size.any():
        _, my0, _, my1 = _bbox_of(m_size)
        s_bbox = th / max(1.0, my1 - my0)
    else:
        s_bbox = 1.0

    # 尺度初值两种口径：
    # * area   —— 面积比 sqrt，对轮廓上多出的一小块（披风、影子）不敏感，
    #             但**手臂伸出/收起会显著改变面积**，clay 侧视图会被算小；
    # * height —— 包围盒高度比，躯干高度稳定，但对头顶/脚尖多出 30px 敏感。
    # 默认 area 保持旧行为；T-pose 侧视手臂长度变化时用 height 更稳。
    s0 = s_bbox
    if (scale_mode == "area" and ref is not None and ref.any()
            and m_size.any()):
        s0 = float(np.sqrt(ref.sum() / m_size.sum()))

    # IoU 搜索在 1/SS 分辨率上做：1024² 上全画幅逐像素滑窗要几秒，缩到 1/4 后
    # 是百毫秒级；平移精度损失（±4px）远小于 Qwen 本身的构图漂移。
    SS = 4
    S = max(64, size // SS)

    def _place(ms: np.ndarray, x0: int, y0: int) -> np.ndarray:
        mh, mw = ms.shape
        cxs, cys = max(0, -x0), max(0, -y0)
        ex, ey = min(mw, S - x0), min(mh, S - y0)
        canvas = np.zeros((S, S), bool)
        if ex > cxs and ey > cys:
            canvas[y0 + cys:y0 + ey, x0 + cxs:x0 + ex] = ms[cys:ey, cxs:ex]
        return canvas

    def _search(ms, ref_s, bx, by, radius, step):
        best = (-1.0, bx, by)
        for x0 in range(bx - radius, bx + radius + 1, step):
            for y0 in range(by - radius, by + radius + 1, step):
                score = _iou(_place(ms, x0, y0), ref_s)
                if score > best[0]:
                    best = (score, x0, y0)
        return best

    best = None
    scales = [s0]
    if refine and ref is not None:
        scales = [s0 * f for f in (0.94, 0.97, 1.0, 1.03, 1.06)]
    ref_s = None
    if ref is not None:
        ref_s = np.asarray(Image.fromarray(ref.astype(np.uint8) * 255)
                           .resize((S, S), Image.NEAREST)) > 127
    for s in scales:
        sw = max(1, int(round(size * s)))      # 源图先铺满画布，再乘相对缩放
        small = im.resize((sw, sw), Image.LANCZOS)
        m = _mask_of(small)
        ys, xs = np.where(m)
        if not len(xs):
            continue
        # 初值：「底边贴齐目标框底边 + 水平居中」
        dx = int(round((tx0 + tx1) / 2 - (xs.min() + xs.max()) / 2))
        dy = int(round(ty1 - ys.max() - 1))
        if ref_s is None:
            best = (s, dx, dy, 1.0, 1.0)
            break
        ms = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                        .resize((max(1, sw // SS), max(1, sw // SS)),
                                Image.NEAREST)) > 127
        # 核心对齐：Qwen 侧视图里细长的手臂/头发长度变化很大，全轮廓 IoU
        # 会被它们主导（实测 v03 只有 0.40）。腐蚀后只剩躯干核心，尺度/平移
        # 估计更稳；最终输出仍用未腐蚀的原图，只是变换参数由核心决定。
        if core_iou:
            kernel = np.ones((3, 3), np.uint8)
            rc = cv2.erode(ref_s.astype(np.uint8), kernel,
                           iterations=2).astype(bool)
            mc = cv2.erode(ms.astype(np.uint8), kernel,
                           iterations=2).astype(bool)
            use_ref, use_ms = (rc, mc) if rc.any() and mc.any() else (ref_s, ms)
        else:
            use_ref, use_ms = ref_s, ms
        bx, by = int(round(dx / SS)), int(round(dy / SS))
        # 两级搜索：粗扫 ±8% 画幅（Qwen 的构图漂移实测能到 0.1 画幅），
        # 再在粗解附近 ±1 细修。
        coarse = max(1, S // 64)
        score, x0, y0 = _search(use_ms, use_ref, bx, by, max(4, S // 12), coarse)
        score, x0, y0 = _search(use_ms, use_ref, x0, y0, coarse, 1)
        if best is None or score > best[3]:
            # iou_full 用全轮廓重算，保证不同对齐口径之间可比
            best = (s, x0 * SS, y0 * SS, score,
                    _iou(_place(ms, x0, y0), ref_s))
    if best is None:
        raise RuntimeError("相似变换对齐失败")

    s, dx, dy, score, iou_full = best
    sw = max(1, int(round(size * s)))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(im.resize((sw, sw), Image.LANCZOS), (dx, dy))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    am = _mask_of(canvas)
    return {"src": str(src), "transparent": round(float(1 - am.mean()), 3),
            "source_box": [sx0, sy0, sx1, sy1],
            "target_box": [dx, dy, dx + sw, dy + sw],
            "scale": round(float(s), 4), "iou": round(float(score), 4),
            "iou_full": round(float(iou_full), 4),
            "iou_mode": "core" if core_iou else "full",
            "align": "similarity", "scale_mode": scale_mode, "split": split}


def make_frame_aligned_view(src: str | Path, dst: str | Path,
                            size: int = 1024) -> Dict[str, Any]:
    """「已经同框」的视图：只等比铺满画布，**不裁 bbox、不重排**。

    引导生成的图（`tools/qwen_guided_views.py`）就是这一类：它是用同一台正交
    相机渲的指引图喂出来的，模型在图里的位置/比例已经与 ProjUV 一致。此时再走
    一次 bbox 裁剪重排反而是灾难 —— 侧视图的 bbox 只有身高的 1/8 宽，把整张方图
    塞进去会被横向压成一条脏带（这正是早先多相机侧面斑驳的来源之一）。
    """
    im = Image.open(src).convert("RGBA")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    alpha = np.asarray(im)[:, :, 3]
    return {"src": str(src), "transparent": round(float((alpha < 20).mean()), 3),
            "source_box": [0, 0, im.width, im.height],
            "target_box": [0, 0, size, size], "aligned": True}


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92,
                    np.power((x + 0.055) / 1.055, 2.4))


def read_base_atlas(mesh_path: str | Path, size: int) -> Optional[np.ndarray]:
    """读输入 GLB 自带的 baseColorTexture，作为未覆盖 texel 的底图。

    PNG 是 sRGB、行从上到下；bake 结果是线性、行从下到上 —— 两者都要转换，
    否则侧背颜色会偏暗且上下颠倒。
    """
    import trimesh
    s = trimesh.load(str(mesh_path), force="scene")
    for g in s.geometry.values():
        mat = getattr(g.visual, "material", None)
        img = getattr(mat, "baseColorTexture", None)
        if isinstance(img, Image.Image):
            arr = np.asarray(img.convert("RGB").resize((size, size),
                                                       Image.LANCZOS),
                             dtype=np.float32) / 255.0
            return np.flipud(_srgb_to_linear(arr)).astype(np.float32)
    return None


def native_atlas_size(mesh_path: str | Path) -> Optional[int]:
    """输入 GLB 自带 baseColorTexture 的边长；没有就 None。"""
    try:
        wh = glbpatch.image_size(mesh_path)
    except (ValueError, KeyError, IndexError):
        return None
    return max(wh) if wh else None


def prepare_atlas_for_patch(atlas_png: str | Path, mesh_path: str | Path,
                            size: int) -> Path:
    """把 RGB 图集转成可替换进原 GLB 的 PNG。

    原贴图带 alpha（alphaMode 不是 OPAQUE，比如发片/镂空）时必须**保留原 alpha
    通道**，否则直接换成 RGB 会把镂空填实。缩放用最近邻 —— alpha 是硬边，
    插值会造出半透明脏边。
    """
    s = trimesh.load(str(mesh_path), force="scene")
    for g in s.geometry.values():
        mat = getattr(g.visual, "material", None)
        img = getattr(mat, "baseColorTexture", None)
        if not isinstance(img, Image.Image):
            continue
        atlas = Image.open(atlas_png).convert("RGB")
        if atlas.size != (size, size):
            atlas = atlas.resize((size, size), Image.LANCZOS)
        mode = img.mode
        alpha_mode = str(getattr(mat, "alphaMode", "") or "").upper()
        if "A" in mode and alpha_mode in ("MASK", "BLEND"):
            a = img.getchannel("A").resize((size, size), Image.NEAREST)
            atlas = atlas.convert("RGBA")
            atlas.putalpha(a)
        out = Path(atlas_png).with_name(f"{Path(atlas_png).stem}_rgb{size}.png")
        atlas.save(out)
        return out
    return Path(atlas_png)


def _fill_holes(acc: np.ndarray, wsum: np.ndarray) -> np.ndarray:
    """多尺度邻域均值扩散：小洞用近邻，大洞用更大范围均值，不用 cv2.inpaint
    （后者对成片空洞会糊成一片）。"""
    out = np.zeros_like(acc)
    filled = wsum > 1e-4          # 单独维护 mask，别用「颜色是否为 0」判断（黑贴图会误判）
    out[filled] = acc[filled] / wsum[filled, None]
    for k in (3, 7, 15, 31, 61, 121, 241):
        if filled.all():
            break
        num = cv2.boxFilter(acc, -1, (k, k), normalize=False)
        den = cv2.boxFilter(wsum, -1, (k, k), normalize=False)[..., None]
        cand = num / np.maximum(den, 1e-6)
        take = (~filled) & (den[..., 0] > 1e-4)
        out[take] = cand[take]
        filled |= take
    # 仍空的地方用全图均值兜底，避免黑洞
    if not filled.all():
        mean = out[filled].mean(0) if filled.any() else np.array([0.5] * 3, np.float32)
        out[~filled] = mean
    return out


def verify_baked_views(work_dir: Path, names: Sequence[str]) -> List[str]:
    """合并前核对 Blender 实际烘出的视图数。

    Blender 的 `-b -P` 里 Python 抛异常时仍可能以 0 退出，宿主侧
    ``subprocess.run(check=True)`` 抓不到；曾出现 8 机位因 UV layer 上限
    只烘出 7 个、却「成功」合并成 84.6% 覆盖贴图的情况。这里强制补齐。
    """
    vdir = Path(work_dir) / "views"
    got = sorted(p.name[:-len("_color.npy")]
                 for p in vdir.glob("*_color.npy"))
    expected = set(names)
    missing = sorted(expected - set(got))
    extra = sorted(set(got) - expected)
    if missing:
        raise RuntimeError(
            f"Blender 只烘了 {len(got)}/{len(expected)} 个视图，缺 {missing}；"
            f"看 Blender 日志里的 UV layer 上限 / 材质错误。"
            f"（多余的缓存：{extra}）")
    return got


def merge(work_dir: Path, size: int,
          base: Optional[np.ndarray] = None,
          blend: str = "weighted",
          detail: float = 0.0,
          detail_sigma: float = 2.0) -> Dict[str, Any]:
    vdir = work_dir / "views"
    colors, weights = [], []
    for p in sorted(vdir.glob("*_color.npy")):
        key = p.name[:-len("_color.npy")]
        wp = vdir / f"{key}_weight.npy"
        if wp.is_file():
            colors.append(np.load(p))
            weights.append(np.load(wp))
    if not colors:
        raise RuntimeError(f"{vdir} 里没有 bake 结果")
    C = np.stack(colors)                 # (V,S,S,3)
    W = np.stack(weights)                # (V,S,S)
    acc = (C * W[..., None]).sum(0)
    wsum = W.sum(0)
    if blend == "winner":
        # 没有 raycast 遮挡剔除时，多相机加权平均会把不同视角的颜色混成脏色。
        # 只取权重最大的那个相机，表面颜色保持单一来源，接缝更干净。
        best = W.argmax(0)
        idx = np.broadcast_to(best[None, :, :, None], (1, size, size, 3))
        atlas = np.take_along_axis(C, idx, axis=0)[0].astype(np.float32)
        filled = W.max(0) > 1e-4
    else:
        atlas = np.zeros((size, size, 3), np.float32)
        filled = wsum > 1e-4
        atlas[filled] = acc[filled] / wsum[filled, None]
    atlas = np.where(filled[..., None], atlas, 0).astype(np.float32)
    if base is not None:
        # 有底图就用底图兜底（侧背保留原生链贴图），不做全图扩散
        atlas[~filled] = base[~filled]
        fallback = "base"
    else:
        # 只用扩散填**空洞**；不能整体覆盖，否则 winner 的单视角颜色
        # 会被悄悄换回加权平均（单测 test_merge_winner_* 钉住）。
        diffuse = _fill_holes(acc, wsum)
        atlas[~filled] = diffuse[~filled]
        fallback = "diffuse"
    if detail > 0 and base is not None and base.shape == atlas.shape:
        # 频率分离：投影的颜色来自原图（忠实但要放大 → 糊），高频材质细节
        # （金属浮雕、布料纹理）从原生贴图借回来。两者共用同一套 UV，
        # 像素天然对齐，所以不会出现「双边缘」。
        hp = base - cv2.GaussianBlur(base, (0, 0), detail_sigma)
        # 只在投影覆盖区加，且边缘羽化，避免 UV 岛边界出现硬缝
        wfeather = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 3.0)
        atlas = np.clip(atlas + detail * hp * wfeather[..., None], 0.0, 1.0).astype(np.float32)
        detail_applied = detail
    else:
        detail_applied = 0.0
    srgb = (np.clip(_linear_to_srgb(atlas), 0, 1) * 255).astype(np.uint8)
    # bake 的 pixels 从左下角开始；PIL 第一行是上方，必须翻转
    srgb = np.flipud(srgb)
    out_png = work_dir / "atlas.png"
    Image.fromarray(srgb, "RGB").save(out_png)
    # 覆盖 mask 单独存一份：排查「哪块是投影来的、哪块是原生兜底」全靠它
    Image.fromarray((np.flipud(filled.astype(np.uint8)) * 255), "L").save(
        work_dir / "mask.png")
    return {"atlas": str(out_png), "views": len(colors),
            "coverage": float(filled.mean()), "fallback": fallback,
            "blend": blend, "weight_max": float(wsum.max()),
            "detail": detail_applied}


def run(mesh: str | Path, out: str | Path, views: List[Dict[str, Any]],
        *, size: int = 0, exponent: float = 3.0, margin: int = 4,
        decimate: int = 0, engine: str = "CYCLES", unwrap: str = "keep",
        fill: str = "base", blend: str = "weighted", detail: float = 0.0,
        occlude: bool = True, ortho: float = 0.0,
        work_dir: Optional[str | Path] = None, reproject: bool = False,
        verbose: bool = True) -> Dict[str, Any]:
    """views: [{"name","image","azimuth","elevation"?}]，image 是原始视图路径。

    size=0 → 自动：keep 模式跟原生贴图同边长（保住原有纹素密度），
    其余 1024。显式给值则按给的来。
    ``ortho`` 必须与 guidance 渲染用的一致（裁满画幅时一起改）。
    """
    ortho_eff = float(ortho) if ortho else ORTHO
    mesh, out = Path(mesh).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if size <= 0:
        native = native_atlas_size(mesh) if unwrap == "keep" else None
        size = native or 1024
        if verbose:
            print(f"  图集边长自动 = {size}"
                  f"{'（跟原生贴图一致）' if native else '（无原生贴图，取缺省）'}"
                  f"，缓存目录里若已有别的尺寸的 npy 请加 --size 或清缓存",
                  flush=True)
    work = Path(work_dir).resolve() if work_dir else ROOT / ".tmp" / f"bake_{mesh.stem}"
    work.mkdir(parents=True, exist_ok=True)
    blender = blender_exe()
    reusing = reproject and (work / "views").is_dir()
    if reusing:
        cfg_path = work / "bake_config.json"
        if cfg_path.is_file():
            cached = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cached.get("size") and int(cached["size"]) != size:
                raise RuntimeError(
                    f"缓存是 {cached['size']}² 的 bake 结果，和这次 size={size} 不一致；"
                    f"换 --work-dir 或删掉 {work}")
            if cached.get("ortho") and abs(float(cached["ortho"]) - ortho_eff) > 1e-3:
                raise RuntimeError(
                    f"缓存是 ortho={cached['ortho']} 的 bake 结果，和这次 "
                    f"ortho={ortho_eff} 不一致；画幅变了必须重烘，不能 Reproject")

    # --- 1. 严格对齐视图 ---
    if not reusing:
        V = normalized_vertices(mesh)
        cfg_views = {}
        for v in views:
            dst = work / "views" / f"{v['name']}.png"
            how = v.get("align") or "bbox"
            if how == "frame":
                info = make_frame_aligned_view(v["image"], dst, size)
            else:
                bbox = projection_bbox(V, float(v.get("azimuth", 0)),
                                       float(v.get("elevation", 0)),
                                       ortho=ortho_eff)
                if how == "similarity":
                    info = make_similarity_aligned_view(
                        v["image"], dst, bbox, size, v.get("ref_mask"),
                        scale_mode=v.get("align_scale", "area"),
                        core_iou=bool(v.get("core_iou", False)))
                else:
                    info = make_aligned_view(v["image"], dst, bbox, size)
            cfg_views[v["name"]] = {
                "image": str(dst.resolve()),
                "azimuth": float(v.get("azimuth", 0)),
                "elevation": float(v.get("elevation", 0)),
            }
            if verbose:
                if how == "frame":
                    msg = "已同框"
                elif how == "similarity":
                    msg = (f"相似变换 缩放 {info['scale']:.3f} "
                           f"IoU {info['iou']:.3f}"
                           + ("（并排双主体已切除多余那个）" if info.get("split") else "")
                           + f" → {info['target_box']}")
                else:
                    msg = (f"模型 bbox {tuple(round(x, 3) for x in bbox)} "
                           f"→ 目标 {info['target_box']}")
                print(f"  对齐 {v['name']}: {msg}  透明占比 {info['transparent']}",
                      flush=True)
                if info["transparent"] < 0.02:
                    # 整幅不透明 = 没抠图。此时 alpha 包围盒 == 整张图，任何对齐都
                    # 退化成「整幅缩放」，投影会糊成一片且不报错。
                    print(f"  ⚠ 视图 {v['name']} 没有透明通道：先跑 "
                          f"tools/matte.py，否则对齐/权重都会失效", flush=True)
        cfg = {"mesh": str(mesh), "work_dir": str(work), "views": cfg_views,
               "size": size, "ortho": ortho_eff, "exponent": exponent,
               "margin": margin, "decimate": decimate, "engine": engine,
               "unwrap": unwrap, "occlude": occlude}
        cfg_path = work / "bake_config.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        # --- 2. Blender 表面投影 + bake ---
        if verbose:
            print(f"  Blender bake {len(views)} 个相机（engine={engine}）…", flush=True)
        # 清掉上一次的 npy：merge() 会 glob 目录里所有 *_color.npy，
        # 残留的旧视图会被混进本次合并，而且会掩盖「少烘了」的错误。
        vdir = work / "views"
        for stale in (list(vdir.glob("*_color.npy"))
                      + list(vdir.glob("*_weight.npy"))):
            stale.unlink()
        subprocess.run([str(blender), "-b", "-P",
                        str(ROOT / "tools" / "blender_bake_views.py"),
                        "--", str(cfg_path)], check=True)
        got = verify_baked_views(work, [v["name"] for v in views])
        if verbose:
            print(f"  Blender 已烘 {len(got)}/{len(views)} 个视图", flush=True)
    elif verbose:
        print("  复用已烘焙视图（Reproject）", flush=True)

    # --- 3. 合并 ---
    # base 兜底要求 UV 没变（底图与 BakeUV 同坐标系）；重新 unwrap 时只能扩散填充
    base = None
    if fill == "base" and unwrap == "keep":
        base = read_base_atlas(mesh, size)
        if base is None and verbose:
            print("  输入 GLB 没有 baseColorTexture，改用扩散填充", flush=True)
    elif fill == "base" and verbose:
        print("  unwrap != keep：底图与 BakeUV 不同坐标系，改用扩散填充", flush=True)
    merged = merge(work, size, base, blend, detail)
    # 真实覆盖：全图数字会被 bake margin 的「岛外扩张色」虚高，必须分开看
    try:
        from . import uvstats
        cov = np.asarray(Image.open(work / "mask.png").convert("L")) > 127
        uv_src = work / "baked_uv.glb"
        if not uv_src.is_file():
            uv_src = mesh          # keep 模式：直接用输入 GLB 的 UV
        isl = uvstats.island_mask(uv_src, size)
        st = uvstats.coverage_stats(cov, isl)
        merged.update(st)
        if verbose and st["uv_usage"] is not None:
            print(f"  UV 岛占用 {st['uv_usage']:.1%}，"
                  f"岛内有效覆盖 {st['coverage_uv']:.1%}，"
                  f"岛外虚高 {st['coverage_ghost']:.1%}", flush=True)
            if (margin >= 16 and st["coverage_ghost"] is not None
                    and st["coverage_ghost"] > st["coverage_uv"]):
                print("  ⚠ margin 较大：新增覆盖大多落在 UV 岛外空白 texel，"
                      "渲染时不采样，不等于画质提升", flush=True)
    except (OSError, ValueError, KeyError) as e:
        if verbose:
            print(f"  UV 覆盖统计跳过（不致命）：{e}", flush=True)
    if verbose:
        print(f"  合并完成：覆盖 {merged['coverage']:.1%}，"
              f"混合 {merged['blend']}，空洞兜底 {merged['fallback']}，"
              f"借原生高频 {merged['detail']}，atlas {merged['atlas']}", flush=True)

    # --- 4. 贴回并导出 ---
    if unwrap == "keep":
        # UV 没变 → 只换 baseColorTexture 的字节，网格原封不动
        png = prepare_atlas_for_patch(merged["atlas"], mesh, size)
        info = glbpatch.patch_base_color(mesh, png, out)
        if verbose:
            print(f"  二进制换贴图：{info['src_size'] / 1e6:.1f}MB → "
                  f"{info['dst_size'] / 1e6:.1f}MB（网格字节未改）", flush=True)
        return {**merged, "path": str(out), "work_dir": str(work),
                "export": "glbpatch", "out_size": info["dst_size"]}

    uv_glb = work / "baked_uv.glb"
    if not uv_glb.is_file():
        raise RuntimeError(f"缺中间网格 {uv_glb}")
    cfg2 = {"mesh": str(uv_glb), "atlas": str(Path(merged["atlas"]).resolve()),
            "out": str(out), "roughness": 0.85}
    cfg2_path = work / "apply_config.json"
    cfg2_path.write_text(json.dumps(cfg2, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    subprocess.run([str(blender), "-b", "-P",
                    str(ROOT / "tools" / "blender_apply_atlas.py"),
                    "--", str(cfg2_path)], check=True)
    return {**merged, "path": str(out), "work_dir": str(work),
            "export": "blender", "out_size": out.stat().st_size}
