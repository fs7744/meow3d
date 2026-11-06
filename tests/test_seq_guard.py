"""顺序生成的「两格拼图」卫士单测。

`tools/qwen_guided_views.py --two-images` 会把参考图和彩图一起喂给 Qwen，
实测会时不时输出「左参考 / 右结果」拼图。那张图整幅投影后，灰色素模的颜色
会糊到模型上（2026-11-06 满身白灰斑的真凶）。判据不是数连通域（两个小人
手部相连会合并成一个），而是「前景有没有跑到网格剪影之外」。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_seq():
    spec = importlib.util.spec_from_file_location(
        "run_sequential", ROOT / "tools" / "run_sequential.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SEQ = _load_seq()


def _silhouette(p: Path, w: int, h: int, x0: int, x1: int) -> Path:
    a = np.zeros((h, w), np.uint8)
    a[:, x0:x1] = 255
    Image.fromarray(np.dstack([np.full((h, w, 3), 255, np.uint8), a]),
                    "RGBA").save(p)
    return p


def _render(p: Path, w: int, h: int, boxes, bg=(210, 210, 210),
            fg=(190, 40, 40)) -> Path:
    a = np.full((h, w, 3), bg, np.uint8)
    for x0, x1 in boxes:
        a[:, x0:x1] = fg
    Image.fromarray(a, "RGB").save(p)
    return p


def test_single_figure_inside_silhouette_passes(tmp_path: Path):
    sil = _silhouette(tmp_path / "sil.png", 120, 100, 25, 95)
    gen = _render(tmp_path / "gen.png", 120, 100, [(32, 88)])
    SEQ._collage_guard(gen, sil, "v00")     # 不抛异常


def test_two_figure_collage_is_rejected(tmp_path: Path):
    sil = _silhouette(tmp_path / "sil.png", 120, 100, 25, 95)
    # 左右各一只，且有大量像素落在剪影外 → 拼图
    gen = _render(tmp_path / "gen.png", 120, 100, [(2, 42), (78, 118)])
    with pytest.raises(RuntimeError, match="剪影之外"):
        SEQ._collage_guard(gen, sil, "v00")


def test_silhouette_from_render(tmp_path: Path):
    """白模没有 UV，首图剪影只能从素模渲染反推。"""
    clay = _render(tmp_path / "clay.png", 160, 160, [(50, 110)],
                   bg=(200, 200, 200), fg=(235, 235, 235))
    out = SEQ._silhouette_from_render(clay, tmp_path / "alpha.png")
    a = np.array(Image.open(out).convert("RGBA"))[:, :, 3]
    assert a.shape == (160, 160)
    assert (a[:, 50:110] > 128).mean() > 0.9
    assert (a[:, :20] > 128).mean() < 0.05


def test_seq_prompt_preserves_canvas():
    """顺序轮必须明确「保留已贴图区域、只补品红」，否则会整幅重绘。"""
    assert "Preserve image 2's existing textured regions" in SEQ.SEQ_PROMPT
    assert "replace every solid magenta region" in SEQ.SEQ_PROMPT


def test_protect_context_keeps_non_magenta_pixels():
    """顺序轮必须等价于 inpaint：mask 外像素原样保留，不能被 Qwen 重画变暗。"""
    ctx = np.full((60, 60, 3), (30, 120, 60), np.uint8)
    ctx[10:30, 10:30] = (255, 0, 255)          # 品红 = 待补
    gen = np.full((60, 60, 3), (200, 10, 10), np.uint8)
    c = Image.fromarray(ctx, "RGB")
    g = Image.fromarray(gen, "RGB")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cp, gp, dp = (Path(td) / "ctx.png", Path(td) / "gen.png",
                      Path(td) / "out.png")
        c.save(cp); g.save(gp)
        st = SEQ._protect_context(gp, cp, dp)
        out = np.array(Image.open(dp).convert("RGB"))
    assert st["edited"] > 0
    assert tuple(out[40, 40]) == (30, 120, 60), "非品红区必须保留 ctx 原色"
    assert tuple(out[20, 20]) == (200, 10, 10), "品红区取生成图"


def test_protect_context_keeps_non_magenta_pixels():
    """顺序轮必须等价于 inpaint：mask 外像素原样保留，不能被 Qwen 重画变暗。"""
    ctx = np.full((60, 60, 3), (30, 120, 60), np.uint8)
    ctx[10:30, 10:30] = (255, 0, 255)          # 品红 = 待补
    gen = np.full((60, 60, 3), (200, 10, 10), np.uint8)
    c = Image.fromarray(ctx, "RGB")
    g = Image.fromarray(gen, "RGB")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cp, gp, dp = (Path(td) / "ctx.png", Path(td) / "gen.png",
                      Path(td) / "out.png")
        c.save(cp); g.save(gp)
        st = SEQ._protect_context(gp, cp, dp)
        out = np.array(Image.open(dp).convert("RGB"))
    assert st["edited"] > 0
    assert tuple(out[40, 40]) == (30, 120, 60), "非品红区必须保留 ctx 原色"
    assert tuple(out[20, 20]) == (200, 10, 10), "品红区取生成图"
