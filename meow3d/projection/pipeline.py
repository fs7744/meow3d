"""投影贴图主管线：多视图 → 加权混合 → UV 补洞 → 导出带贴图 GLB。

流程（全部按公开思路自己实现，未复制 StableGen 的 GPL 源码）：

1. `build_uv_map`   每个 texel 属于哪个面 / 对应 3D 点与法线（最贵，缓存 npz）
2. `make_cameras`   环绕相机
3. `ViewProvider`   每相机视图（正面原图直贴 + 其余几何重投影；可插拔扩散模型）
4. `visibility_weights`  cos^k 加权 + 遮挡剔除 + DPPG 深度边界抑制
5. `blend`          按权重累加、归一化；未覆盖 UV 用底图或 inpaint 补
6. `export`         trimesh 写 GLB

`reuse_views=True` 时跳过 3、从缓存读视图 —— 改混合参数只重算 4/5，
这就是 Reproject（不必重新生成视图）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import trimesh
from PIL import Image

from . import geometry, views as viewsmod


def _load_base_atlas(mesh: Any, size: int) -> Optional[np.ndarray]:
    m = getattr(mesh.visual, "material", None)
    img = getattr(m, "baseColorTexture", None)
    if img is None:
        return None
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS))
    return arr.astype(np.float32) / 255.0


def _sample(view: viewsmod.View, mapx: np.ndarray, mapy: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.remap(view.image, mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    msk = cv2.remap(view.mask, mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, np.clip(msk, 0, 1)


def _blend(uvm: geometry.UVMap, cams: List[geometry.Camera],
           prepared: List[Optional[viewsmod.View]], *, exponent: float,
           discard_deg: float, dppg: bool, dppg_margin: int,
           verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    S = uvm.size
    acc = np.zeros((S, S, 3), np.float32)
    wsum = np.zeros((S, S), np.float32)

    for i, (cam, view) in enumerate(zip(cams, prepared)):
        if view is None:
            continue
        scr = cam.screen(uvm.points)
        mapx = (scr[..., 0] * S).astype(np.float32)
        mapy = (scr[..., 1] * S).astype(np.float32)
        color, vmask = _sample(view, mapx, mapy)
        w = geometry.visibility_weights(
            uvm, cam, exponent=exponent, discard_deg=discard_deg,
            dppg=dppg, dppg_margin=dppg_margin)
        w = w * vmask
        acc += color * w[..., None]
        wsum += w
        if verbose:
            # 按「有面的 texel」算，UV 图集利用率低是 xatlas 的正常现象
            print(f"  [{cam.name}] 有效贡献 {(w > 1e-4).sum() / max(1, uvm.valid.sum()):.1%}",
                  flush=True)

    # DPPG 抑制过头会留下空洞 → coverage rescue：用不带 DPPG 的权重补一遍
    hole = uvm.valid & (wsum < 1e-4)
    if dppg and hole.any():
        if verbose:
            print(f"  coverage rescue：{hole.mean():.1%} texel 用无 DPPG 权重补", flush=True)
        for cam, view in zip(cams, prepared):
            if view is None or not hole.any():
                continue
            scr = cam.screen(uvm.points)
            mapx = (scr[..., 0] * S).astype(np.float32)
            mapy = (scr[..., 1] * S).astype(np.float32)
            color, vmask = _sample(view, mapx, mapy)
            w = geometry.visibility_weights(
                uvm, cam, exponent=exponent, discard_deg=discard_deg,
                dppg=False) * vmask * hole
            acc += color * w[..., None]
            wsum += w

    covered = wsum > 1e-4
    atlas = np.where(covered[..., None], acc / (wsum[..., None] + 1e-6), 0)
    return np.clip(atlas, 0, 1), wsum


def _fill_holes(atlas: np.ndarray, wsum: np.ndarray, uvm: geometry.UVMap,
                base: Optional[np.ndarray]) -> np.ndarray:
    hole = uvm.valid & (wsum <= 1e-4)
    if not hole.any():
        return atlas
    if base is not None:
        out = atlas.copy()
        out[hole] = base[hole]
        return out
    bgr = (atlas * 255).astype(np.uint8)
    mask = (hole.astype(np.uint8) * 255)
    filled = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    out = atlas.copy()
    out[hole] = filled.astype(np.float32)[hole] / 255.0
    return out


def _replace_front(uvm: geometry.UVMap, cam: geometry.Camera,
                   view: viewsmod.View, base: np.ndarray, *, exponent: float,
                   discard_deg: float, dppg: bool, dppg_margin: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """保守模式：只在正面相机能看到、且原图有像素的地方做替换，其余原样保留。

    没有 2D 生成模型时，「把 UV 图集重投影成侧背视图」不会增加任何信息，
    只会因 UV 岛碎片化而错位。真正稳的是 GD-Head / Headshot 3 那套：
    可见区直接取原图像素，不可见区保持原样。
    """
    S = uvm.size
    scr = cam.screen(uvm.points)
    color, vmask = _sample(view, (scr[..., 0] * S).astype(np.float32),
                           (scr[..., 1] * S).astype(np.float32))
    w = geometry.visibility_weights(uvm, cam, exponent=exponent,
                                    discard_deg=discard_deg, dppg=dppg,
                                    dppg_margin=dppg_margin)
    pool = w[uvm.valid & (w > 0)]
    ref = float(np.quantile(pool, 0.95)) if pool.size > 100 else max(1e-6, w.max())
    strength = np.clip(w / max(ref, 1e-6), 0, 1) ** 0.7
    strength *= vmask
    alpha = np.clip(strength, 0, 1)[..., None]
    atlas = np.clip(base * (1 - alpha) + color * alpha, 0, 1)
    return atlas, strength


def run(mesh_path: str | Path, reference: str | Path, out_path: str | Path,
        *, size: int = 1024, cameras: int = 8, exponent: float = 3.0,
        discard_deg: float = 90.0, dppg: bool = True, dppg_margin: int = 3,
        mode: str = "replace", provider: Optional[viewsmod.ViewProvider] = None,
        views_dir: Optional[str | Path] = None, fill: str = "base",
        cache_dir: Optional[str | Path] = None, reuse_views: bool = False,
        verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    mesh_path, out_path = Path(mesh_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = Path(cache_dir) if cache_dir else (out_path.parent / ".cache")
    stem = mesh_path.stem
    uv_cache = cache / f"{stem}_uv{size}.npz"
    view_dir = cache / f"{stem}_views{size}"
    view_dir.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(str(mesh_path), force="scene")
    mesh = list(scene.geometry.values())[0]
    base = _load_base_atlas(mesh, size)
    if getattr(mesh.visual, "uv", None) is None:
        raise RuntimeError(
            f"{mesh_path.name} 没有 UV。请先用 tools/uv_unwrap.py 展开 UV，"
            "或用带纹理的 GLB 作为输入（只替换贴图）。")

    if verbose:
        print(f"网格 {len(mesh.vertices)} 顶点 / {len(mesh.faces)} 面，UV 图集 {size}²", flush=True)
    uvm = geometry.build_uv_map(mesh, size, uv_cache, verbose)
    cams, _ = geometry.make_cameras(uvm.points[uvm.valid], cameras)

    # 视图：可复用缓存 → Reproject
    use_cache = reuse_views and all(
        (view_dir / f"{i:02d}_img.npy").is_file() for i in range(len(cams)))
    prepared: List[Optional[viewsmod.View]] = []
    if use_cache:
        if verbose:
            print("  Reproject：复用已缓存的相机视图，只重算混合", flush=True)
        for i in range(len(cams)):
            img = np.load(view_dir / f"{i:02d}_img.npy")
            msk = np.load(view_dir / f"{i:02d}_msk.npy")
            prepared.append(viewsmod.View(img, msk))
    else:
        if provider is None:
            if mode == "qwen":
                if views_dir is None:
                    raise ValueError("mode=qwen 需要 --views-dir（tools/qwen_views.py 产物）")
                provider = viewsmod.QwenViewProvider(views_dir)
                if verbose:
                    print(f"  视图来源：Qwen 多视角 {views_dir}", flush=True)
            else:
                provider = viewsmod.HybridViewProvider(reference, base_atlas=base)
        prepared = provider.build(uvm, cams, base if base is not None
                                  else np.zeros((size, size, 3), np.float32), size)
        for i, v in enumerate(prepared):
            if v is not None:
                np.save(view_dir / f"{i:02d}_img.npy", v.image.astype(np.float32))
                np.save(view_dir / f"{i:02d}_msk.npy", v.mask.astype(np.float32))
        (view_dir / "params.json").write_text(json.dumps(
            {"reference": str(reference), "provider": type(provider).__name__,
             "size": size, "cameras": cameras}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    empty = np.zeros((size, size, 3), np.float32)
    if mode == "replace":
        if verbose:
            print("  模式 replace：正面原图像素直贴，其余 texel 保持底图", flush=True)
        atlas, wsum = _replace_front(
            uvm, cams[0], prepared[0], base if base is not None else empty,
            exponent=exponent, discard_deg=discard_deg, dppg=dppg,
            dppg_margin=dppg_margin)
    elif mode in ("blend", "qwen"):
        if verbose:
            tag = "Qwen 新视角" if mode == "qwen" else "仅几何重投影（实验性）"
            print(f"  模式 {mode}：完整多相机混合，视图来源 {tag}", flush=True)
        atlas, wsum = _blend(uvm, cams, prepared, exponent=exponent,
                             discard_deg=discard_deg, dppg=dppg,
                             dppg_margin=dppg_margin, verbose=verbose)
        atlas = _fill_holes(atlas, wsum, uvm,
                            base if fill == "base" else None)
    else:
        raise ValueError(f"未知 mode={mode}，可选 replace / blend / qwen")

    img = Image.fromarray((atlas * 255).astype(np.uint8), "RGB")
    uv = np.asarray(mesh.visual.uv, dtype=np.float64).copy()
    out_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                               faces=np.asarray(mesh.faces),
                               visual=trimesh.visual.TextureVisuals(uv=uv, image=img),
                               process=False)
    out_mesh.export(str(out_path), file_type="glb")
    cov = float((wsum > 1e-4).sum() / max(1, uvm.valid.sum()))
    if verbose:
        print(f"  导出 {out_path}  有面 texel 覆盖 {cov:.1%}  "
              f"用时 {time.time() - t0:.1f}s", flush=True)
    return {"path": str(out_path), "coverage": cov,
            "elapsed": round(time.time() - t0, 1),
            "atlas": atlas, "uv_valid": float(uvm.valid.mean())}
