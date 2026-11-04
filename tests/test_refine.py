"""refine 的纯逻辑单测（无 GPU / Blender）。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from meow3d.projection import refine


def _solid(w=64, h=64, color=(0.8, 0.2, 0.1)) -> np.ndarray:
    a = np.zeros((h, w, 3), np.float32)
    a[...] = np.array(color, np.float32)
    return a


def _figure(color=(0.8, 0.2, 0.1), bg=(0.6, 0.6, 0.62), w=64, h=64):
    a = np.zeros((h, w, 3), np.float32)
    a[...] = np.array(bg, np.float32)
    a[16:48, 16:48] = np.array(color, np.float32)
    mask = np.zeros((h, w), bool)
    mask[16:48, 16:48] = True
    return a, mask


def test_background_mask_finds_solid_world_background():
    im, truth = _figure()
    got = refine.background_mask(im)
    # 四角是纯色背景；中间方块应被判为前景
    assert got[32, 32]
    assert not got[0, 0]
    assert got.sum() > 0.5 * truth.sum()


def test_anchor_color_removes_global_cast_but_keeps_structure():
    base, mask = _figure(color=(0.8, 0.2, 0.1))
    # 生成图整体偏亮偏冷：模拟 8 步 Lightning 的「发色变浅、颜色变淡」
    src = base.copy()
    src[mask] = src[mask] * 0.7 + np.array([0.18, 0.20, 0.24], np.float32)
    before = src[mask].mean(0)
    out, stats = refine.anchor_color(src, base, mask)
    after = out[mask].mean(0)
    assert stats["applied"] and stats["pixels"] == int(mask.sum())
    assert np.abs(after - base[mask].mean(0)).max() < 0.02
    assert np.abs(before - base[mask].mean(0)).max() > 0.05
    # mask 外原样保留
    assert np.array_equal(out[~mask], src[~mask])


def test_anchor_color_noop_when_mask_too_small():
    base = _solid()
    mask = np.zeros(base.shape[:2], bool)
    mask[0, 0] = True
    out, stats = refine.anchor_color(base, base * 0.5, mask)
    assert stats["applied"] is False
    assert np.array_equal(out, base)


def test_protect_one_keeps_clean_base_untouched():
    base, mask = _figure()
    # 生成图与底图只有整体色偏，没有新增细节 → detail_gain≈0
    src = np.clip(base * 0.75 + np.array([0.1, 0.12, 0.15], np.float32), 0, 1)
    out, stats = refine.protective_blend(src, base, mask, protect=1.0)
    assert stats["mean_weight"] < 0.02
    assert np.abs(out[mask] - base[mask]).max() < 0.03


def test_protect_equals_full_replacement_at_zero():
    base, mask = _figure()
    src = np.clip(base * 0.75 + 0.1, 0, 1)
    out, stats = refine.protective_blend(src, base, mask, protect=0.0)
    assert stats["mean_weight"] == 1.0
    # 颜色已被锚定回底图，所以结果与 base 接近，但权重确实是 1
    assert stats["max_weight"] == 1.0


def test_detail_gain_releases_blurred_base_only():
    base, mask = _figure()
    base_blur = np.asarray(Image.fromarray(
        (np.clip(base, 0, 1) * 255).astype(np.uint8)
    ).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0
    # 生成图带清晰硬边，底图被糊掉 → 放行权重应明显高于「底图本来清晰」
    g_blur = refine.detail_gain(base, base_blur, mask)
    g_clean = refine.detail_gain(base, base, mask)
    # 平涂大色块仍应保持 0（不该被 Qwen 改写），只有硬边附近放行
    assert g_blur[mask].mean() > g_clean[mask].mean() + 0.05
    assert g_blur[mask].max() > 0.25
    assert g_clean[mask].mean() < 0.01
    assert not g_blur[~mask].any()


def test_save_rgba_roundtrip(tmp_path):
    rgb = _solid()
    alpha = np.zeros(rgb.shape[:2], np.float32)
    alpha[10:20, 10:20] = 1.0
    p = tmp_path / "v.png"
    refine.save_rgba(rgb, alpha, p)
    assert Image.open(p).mode == "RGBA"
    assert np.allclose(refine.load_alpha(p), alpha, atol=0.01)
