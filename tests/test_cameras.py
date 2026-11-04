"""相机布置的纯逻辑单测（不读大 GLB）。"""
from __future__ import annotations

import math

import numpy as np

from meow3d.projection import cameras


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_direction_to_angles_matches_bake_convention():
    # d = (sin az cos el, -cos az cos el, sin el)
    for az, el in [(0, 0), (90, 0), (180, 0), (270, 0), (45, 20)]:
        a, e = math.radians(az), math.radians(el)
        d = np.array([math.sin(a) * math.cos(e),
                      -math.cos(a) * math.cos(e), math.sin(e)])
        got_az, got_el = cameras.direction_to_angles(d, max_elevation=90)
        assert abs(got_az - az) < 0.01, (az, got_az)
        assert abs(got_el - el) < 0.01, (el, got_el)


def test_direction_to_angles_clamps_elevation():
    d = np.array([0.0, -0.2, 0.98])      # 接近正上方
    _, el = cameras.direction_to_angles(d, max_elevation=30)
    assert abs(el) <= 30.01
    # clamp 后仍是单位向量意义上的合法方向（xy 被重新归一化）
    assert el == 30.0 or el == -30.0


def test_kmeans_recovers_four_clusters():
    rng = np.random.default_rng(3)
    centers = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], np.float32)
    n, w = [], []
    for c in centers:
        for _ in range(40):
            v = c + rng.normal(0, 0.08, 3)
            n.append(_unit(v))
            w.append(1.0)
    D = cameras.kmeans_on_sphere(np.asarray(n), np.asarray(w), 4, seed=1)
    assert D.shape == (4, 3)
    # 每个真中心都能找到一个几乎同向的聚类中心
    sim = np.abs(np.asarray(n) @ D.T)
    for c in centers:
        assert (np.asarray(centers) @ D.T).max(1).min() > 0.9


def test_ring_plan_keeps_legacy_behaviour():
    p = cameras.ring_plan(8, [0, 2, 4, 6])
    assert [c["azimuth"] for c in p] == [0, 90, 180, 270]
    assert all(c["elevation"] == 0 for c in p)
    assert [c["name"] for c in p] == ["v00", "v02", "v04", "v06"]


def test_save_load_and_angles_for(tmp_path):
    p = cameras.ring_plan(4, elevation=12)
    f = cameras.save_plan(p, tmp_path / "plan.json")
    got = cameras.load_plan(f)
    assert got == p
    assert cameras.angles_for(got, 2) == (180.0, 12.0)
    assert cameras.angles_for(got, 99) == (0.0, 0.0)


def test_normal_weighted_plan_front_first_and_bounded(monkeypatch):
    # 用合成法线替掉 GLB 读取，避免单测依赖 20MB 网格
    n = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0.5, 0.5, 0.5]], np.float32)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    w = np.ones(len(n), np.float32)
    monkeypatch.setattr(cameras, "face_frame", lambda *a, **k: (n, w))
    plan = cameras.normal_weighted_plan("fake.glb", count=8,
                                        max_elevation=30, seed=2)
    assert len(plan) == 8
    assert plan[0]["azimuth"] == 0.0 and plan[0]["elevation"] == 0.0
    assert all(abs(c["elevation"]) <= 30.01 for c in plan)
    assert all(-0.01 <= c["azimuth"] < 360.01 for c in plan)
    assert len({c["name"] for c in plan}) == 8
    # 其余按 azimuth 升序，方便对照
    azs = [c["azimuth"] for c in plan[1:]]
    assert azs == sorted(azs)


def test_dedupe_removes_near_duplicates():
    out = cameras._dedupe([np.array([1, 0, 0]),
                           np.array([0.999, 0.01, 0]),
                           np.array([0, 1, 0])], min_deg=22)
    assert len(out) == 2
