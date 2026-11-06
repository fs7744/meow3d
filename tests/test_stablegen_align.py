"""前半段按 StableGen 默认值对齐的回归测试（纯逻辑，不加载模型）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import api, graph  # noqa: E402


def _wf():
    return graph.load_workflow(api.DEFAULT_WORKFLOW)


def _first(wf, node_type):
    return next(n for n in wf["nodes"] if n["type"] == node_type)


def _primitive(wf, value):
    return next(n for n in wf["nodes"]
                if n["type"] == "PrimitiveInt" and n["widgets_values"][0] == value)


def test_generate_defaults_are_rolled_back():
    """防止再把已证伪的参数对齐设回默认。"""
    import inspect
    sig = inspect.signature(api.generate)
    assert sig.parameters["align_stablegen"].default is False
    assert sig.parameters["bg_removal"].default == "skip"


def test_official_workflow_original_values():
    """官方示例把节点自带的 StableGen 等价默认值改低了。"""
    wf = _wf()
    assert _primitive(wf, 500000) is not None       # 50 万面
    assert _primitive(wf, 2048) is not None         # 2048 图集
    assert _first(wf, "Trellis2PreProcessImage")["widgets_values"][1] is False
    tex = _first(wf, "Trellis2MeshTexturing")["widgets_values"]
    assert tex[3] == 5                             # texture_guidance_strength
    assert tex[7] == 2048                          # texture_size


def test_align_applies_stablegen_defaults():
    wf = _wf()
    kw = api.stablegen_align(wf)
    tex_id = int(_first(wf, "Trellis2MeshTexturing")["id"])
    rec_id = int(_first(wf, "Trellis2ReconstructMeshWithQuad")["id"])
    pre_id = int(_first(wf, "Trellis2PreProcessImage")["id"])
    assert kw[tex_id]["texture_size"] == 4096
    assert kw[tex_id]["texture_guidance_strength"] == 3.0
    assert kw[rec_id]["remove_inner_faces"] is False
    # 抠图在宿主侧用 BiRefNet 完成，工作流内不再二次抠
    assert kw[pre_id]["remove_background"] is False
    assert _primitive(wf, 100000) is not None
    assert _primitive(wf, 4096) is not None
    prims = [n["widgets_values"][0] for n in wf["nodes"]
             if n["type"] == "PrimitiveInt"]
    assert 500000 not in prims and 2048 not in prims  # 原值已被替换


def test_align_disabled_is_noop():
    wf = _wf()
    assert api.stablegen_align(wf, enabled=False) == {}
    assert _primitive(wf, 500000) is not None
    assert _primitive(wf, 2048) is not None


def test_align_is_idempotent():
    wf = _wf()
    a = api.stablegen_align(wf)
    b = api.stablegen_align(wf)
    assert a == b
    assert _primitive(wf, 100000) is not None


def test_image_tensor_auto_calls_birefnet(tmp_path, monkeypatch):
    """auto 走 rembg（BiRefNet），skip 沿用输入 alpha。"""
    pytest.importorskip("torch")
    import rembg
    from PIL import Image

    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(p)

    called = []

    def fake_remove(img, session=None):
        called.append(session)
        return Image.new("RGBA", (8, 8), (1, 2, 3, 128))

    monkeypatch.setattr(rembg, "remove", fake_remove)
    monkeypatch.setattr(api, "_matte_session", lambda model: "SESSION")

    t = api.image_tensor(p, bg_removal="auto")
    assert called == ["SESSION"]
    assert tuple(t.shape) == (1, 8, 8, 4)
    assert abs(float(t[0, 0, 0, 3]) - 128 / 255) < 1e-6

    t2 = api.image_tensor(p, bg_removal="skip")
    assert called == ["SESSION"]               # skip 不碰 rembg
    assert float(t2[0, 0, 0, 3]) == 1.0        # 无 alpha → 全不透明
