"""顺序生成（sequential）的纯逻辑单测：ctx 图集的品红规则 + 对齐约束。

不启 Blender、不加载模型，只钉住两条容易再次踩坑的规则：

1. **品红只涂 UV 岛内的未覆盖 texel**。整张「未覆盖」都涂会把岛外空白也涂掉，
   渲染降采样时岛边界的品红会混进角色轮廓（实测出现过「满身红斑点」）。
2. **顺序生成只允许 frame 对齐**。任何 2D 变换都会让生成图与相机像素错位，
   直接导致重影（`similarity` 的老毛病）。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import bake  # noqa: E402


def _load_seq():
    spec = importlib.util.spec_from_file_location(
        "run_sequential", ROOT / "tools" / "run_sequential.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SEQ = _load_seq()


def test_magenta_only_inside_islands():
    atlas = np.full((32, 32, 3), 90, np.uint8)          # 岛外：暗灰
    atlas[8:24, 8:24] = (200, 40, 40)                   # 岛内：红
    islands = np.zeros((32, 32), np.uint8)
    islands[8:24, 8:24] = 1
    mask = np.full((32, 32), 255, np.uint8)             # 默认全覆盖
    mask[10:12, 10:12] = 0                              # 岛内留一个洞

    img = np.array(SEQ._build_ctx_atlas(atlas, mask, islands))
    assert tuple(img[10, 10]) == (255, 0, 255), "岛内未覆盖 texel 必须是品红"
    assert tuple(img[20, 20]) == (200, 40, 40), "岛内已覆盖 texel 不能被改"
    # 岛外即使 mask=0 也不能是品红（否则降采样会污染角色边缘）
    outside = mask.copy()
    outside[:, :8] = 0
    img2 = np.array(SEQ._build_ctx_atlas(atlas, outside, islands))
    assert not np.any(np.all(img2[:, :8] == (255, 0, 255), axis=-1))


def test_outside_islands_filled_by_neighbour_colour():
    """岛外应当被最近的岛色膨胀填满，而不是留黑/留品红。"""
    atlas = np.zeros((24, 24, 3), np.uint8)
    atlas[10:14, 10:14] = (10, 200, 30)
    islands = np.zeros((24, 24), np.uint8)
    islands[10:14, 10:14] = 1
    mask = np.ones((24, 24), np.uint8) * 255
    img = np.array(SEQ._build_ctx_atlas(atlas, mask, islands))
    # 紧邻岛的空白 texel 已吸到岛色
    assert tuple(img[9, 9]) == (10, 200, 30)


def test_bake_views_only_rejects_2d_warp():
    """顺序生成不允许 similarity/bbox 对齐（会直接造成重影）。"""
    with pytest.raises(ValueError):
        bake.bake_views_only("x.glb", ".tmp/never", [
            {"name": "v00", "image": "x.png", "align": "similarity"}], size=64)


def test_sequential_prompt_marks_canvas_as_image2():
    """ctx 是 image 2、参考图是 image 1 —— 编号写反会让模型重画姿势。"""
    assert "image 2's camera viewpoint" in SEQ.SEQ_PROMPT
    assert "Use image 1 only as the identity" in SEQ.SEQ_PROMPT
    assert "solid magenta region" in SEQ.SEQ_PROMPT
