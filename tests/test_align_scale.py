"""similarity 对齐的尺度口径单测（纯逻辑，不需要 Blender/GPU）。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from meow3d.projection import bake

S = 256
# 源图：宽而矮的矩形（面积大、高度小）
SRC_BOX = (98, 158, 58, 198)
# 参考轮廓：窄而高的矩形（面积小、高度大）
REF_BOX = (48, 208, 108, 148)


def _inputs(tmp_path):
    src = tmp_path / "src.png"
    ref = tmp_path / "ref.png"
    a = np.zeros((S, S, 4), np.uint8)
    a[SRC_BOX[0]:SRC_BOX[1], SRC_BOX[2]:SRC_BOX[3]] = [200, 30, 40, 255]
    Image.fromarray(a, "RGBA").save(src)
    r = np.zeros((S, S), np.uint8)
    r[REF_BOX[0]:REF_BOX[1], REF_BOX[2]:REF_BOX[3]] = 255
    Image.fromarray(r, "L").save(ref)
    return src, ref


def test_scale_mode_area_vs_height_gives_different_scale(tmp_path):
    src, ref = _inputs(tmp_path)
    bbox = (0.1, 0.1, 0.9, 0.9)
    ia = bake.make_similarity_aligned_view(
        src, tmp_path / "a.png", bbox, S, ref_mask=ref,
        refine=False, scale_mode="area")
    ih = bake.make_similarity_aligned_view(
        src, tmp_path / "h.png", bbox, S, ref_mask=ref,
        refine=False, scale_mode="height")
    assert ia["scale_mode"] == "area"
    assert ih["scale_mode"] == "height"
    # 面积比 sqrt(6400/8400)≈0.87，高度比 204.8/60≈3.41，必须明显不同
    assert abs(ia["scale"] - ih["scale"]) > 0.5
    assert ia["scale"] < 1.0 < ih["scale"]
    # 非 core 模式时，报告 IoU 与全轮廓 IoU 一致
    assert ia["iou_full"] == ia["iou"]


def test_core_iou_reports_both_scores_and_never_crashes(tmp_path):
    src, ref = _inputs(tmp_path)
    ic = bake.make_similarity_aligned_view(
        src, tmp_path / "c.png", (0.1, 0.1, 0.9, 0.9), S, ref_mask=ref,
        refine=False, scale_mode="height", core_iou=True)
    assert ic["iou_mode"] == "core"
    assert 0.0 <= ic["iou"] <= 1.0
    assert 0.0 <= ic["iou_full"] <= 1.0
    assert (tmp_path / "c.png").is_file()


def test_alignment_output_is_rgba_same_size(tmp_path):
    src, ref = _inputs(tmp_path)
    info = bake.make_similarity_aligned_view(
        src, tmp_path / "o.png", (0.1, 0.1, 0.9, 0.9), S, ref_mask=ref,
        refine=False, scale_mode="height")
    im = Image.open(tmp_path / "o.png")
    assert im.mode == "RGBA" and im.size == (S, S)
    assert info["align"] == "similarity"
