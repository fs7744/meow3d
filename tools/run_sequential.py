"""StableGen v0.5.0 顺序生成（sequential）的逐视图增量实现。

严格照 `stablegen/texturing/generator.py:1848 async_generate` 的主循环：

    for i in 0..N-1:
        if i == 0:  用「参考图 + 结构图」生成（generate_qwen_edit）
        else:       渲 ctx（当前贴图 + 缺失区品红）→ 让 Qwen 只补品红
        投影 img_i → 累积进同一张 BakeUV（project_image(stop_index=i)）

关键事实（都已在源码里核实）：
* ctx 渲染 `export_emit_image`（generator.py:2047）：当前贴图按 Emission 渲出，
  **缺失上下文用 fallback 品红 (1,0,1)，背景也是品红**（`_get_qwen_context_colors`，
  默认 `qwen_context_render_mode='NONE'`）。
* 提示词 `qwen_models.py:169-175`：
  "Preserve its existing textured regions and replace every solid magenta
   region with texture that seamlessly continues the surrounding surface."
* 每轮 `project_image(stop_index=i)` 会**重投 0..i 全部相机**；因为
  `merge()` 把 `views/*.npy` 全量相加，所以逐轮追加 npy 与它等价。

与单轮批量生成的区别（也是之前「更花」的原因）：每轮只看得到**此前的结果**，
不存在 8 张互不一致的整幅重绘。

用法：
    python tools/run_sequential.py out/native_q21_white.glb \
        --reference out/in/asuka_ref.png --out out/asuka_seq2.glb \
        --cameras 8 --atlas-size 2048 --view-size 1024 --steps 20
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d.projection import bake  # noqa: E402

HOST_PY = Path(r"D:\code\2d-game-meow\.venv\Scripts\python.exe")

# StableGen `qwen_models.py:169` 的原文语义（REPLACE_STYLE：canvas=image 2，
# appearance=image 1 —— canvas 是 ctx 渲图，参考图只是外观依据）
SEQ_PROMPT = (
    "Preserve image 2's camera viewpoint, subject orientation, pose, silhouette, "
    "proportions, framing, and background. Use image 1 only as the identity and "
    "appearance reference for the subject. Preserve image 2's existing textured "
    "regions "
    "and replace every solid magenta region with texture that seamlessly continues "
    "the surrounding surface. Do not change the geometry, the outline, or the "
    "position of the character. One single character only, no collage, no split "
    "screen. The camera is {word}."
)

FIRST_PROMPT = (
    "Redraw this character design as a full body game character in an exact T-pose, "
    "matching the depth guide's silhouette and proportions precisely. "
    "CRITICAL POSE CONSTRAINT: the 3D mesh is in a STRICT T-POSE — BOTH arms are "
    "stretched straight out horizontally to the sides, perpendicular to the torso. "
    "This must be preserved. Keep the design of the reference: same clothing, "
    "same colors, same hair, same face. Output ONE single image, one character. "
    "Do NOT put the two images side by side. No collage, no split screen. "
    "Full body from head to feet, even lighting, plain light gray background. "
    "The camera is {word}."
)


def _build_ctx_atlas(atlas: np.ndarray, mask: np.ndarray,
                     islands: np.ndarray | None) -> Image.Image:
    """当前贴图 → ctx 用的图集。

    两件事（都是为了一次正确的重采样）：

    1. **品红只涂 UV 岛内、且确实没被投影到的 texel**。原先整张「未覆盖」
       都涂品红，把岛外空白也涂了；渲染后降采样时，岛边界会把品红混进角色
       轮廓，看起来像满身红斑点。
    2. **岛外先按最近岛色膨胀填满**，这样降采样邻域里不会出现空白/品红，
       采样结果才等于「角色现在的样子」。
    """
    import cv2

    rgb = atlas.astype(np.uint8).copy()
    inside = (islands > 0) if islands is not None else np.ones(mask.shape, bool)
    known = inside.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(16):
        if known.all():
            break
        dil = cv2.dilate(rgb, k)
        nk = cv2.dilate(known.astype(np.uint8), k) > 0
        take = (~known) & nk
        rgb[take] = dil[take]
        known = nk
    need = inside & (mask < 128)
    rgb[need] = (255, 0, 255)
    return Image.fromarray(rgb, "RGB")


def _resize(src: Path, dst: Path, size: int, mode: str) -> None:
    im = Image.open(src).convert(mode)
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    im.save(dst)


def _collage_guard(gen_p: Path, silhouette_p: Path, tag: str) -> None:
    """检测「左参考 / 右结果」两格拼图。

    只数连通域不管用：实测 v00 的两小人**手部相连**成一个整体，照样漏判。
    可靠判据是「前景有没有跑到网格剪影之外」——拼图必然把第二个人物画在
    剪影覆盖不到的地方（2026-11-06 实测：红色参考 + 灰色素模并排）。
    """
    import cv2

    a = np.array(Image.open(gen_p).convert("RGB")).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    fg = (np.abs(a - bg).sum(axis=2) > 40).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    sil = np.array(Image.open(silhouette_p).convert("RGBA"))[:, :, 3] > 96
    if fg.shape != sil.shape:
        fg = cv2.resize(fg, (sil.shape[1], sil.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    total = int(fg.sum())
    if total < 0.01 * fg.size:
        return
    outside = int((fg.astype(bool) & ~sil).sum())
    frac = outside / total
    if frac > 0.45:   # 只在「另一整只人物」这种量级才判拼图
        raise RuntimeError(
            f"{tag} 的前景有 {frac:.0%} 落在网格剪影之外（{gen_p}）："
            f"典型的「参考图 + 结果」两格拼图。")


def _generate_with_guard(cmd: list[str], gen_p: Path, sil_p: Path,
                         tag: str, log: Path, retries: int = 2,
                         reuse: bool = False) -> None:
    """生成 + 拼图检测，失败就换 seed 重试。

    不重试的话，第 3 轮被拦下会白跑前面几轮（每轮约 1 分钟）。
    Qwen 的两格拼图是随机失效，换 seed 基本一次就能过。
    ``reuse`` 为续跑：先检查已有图，不合格才重新生成。
    """
    if reuse and gen_p.is_file():
        try:
            _collage_guard(gen_p, sil_p, tag)
            print(f"  {tag} 复用已有生成图", flush=True)
            return
        except RuntimeError as e:
            print(f"  ⚠ 已有图不合格，重新生成：{e}", flush=True)
            gen_p.unlink(missing_ok=True)
    for attempt in range(retries + 1):
        trial = list(cmd)
        if attempt:
            si = trial.index("--seed") + 1
            trial[si] = str(int(trial[si]) + attempt * 7919)
            gen_p.unlink(missing_ok=True)
        rc = _run(trial, log=log)
        if rc != 0:
            raise RuntimeError(f"{tag} Qwen 生成失败，看 {log}")
        try:
            _collage_guard(gen_p, sil_p, tag)
            if attempt:
                print(f"  {tag} 第 {attempt + 1} 次生成通过拼图检测", flush=True)
            return
        except RuntimeError as e:
            print(f"  ⚠ {e}", flush=True)
            if attempt == retries:
                raise RuntimeError(
                    f"{tag} 连续 {retries + 1} 次生成都是拼图/错位图，"
                    f"换 --seed 或检查提示词") from e
    raise AssertionError("unreachable")


def _silhouette_from_render(render_p: Path, out_p: Path) -> Path:
    """从素模渲染反推网格剪影（白模没有 UV，不能走 ctx 脚本）。"""
    import cv2

    a = np.array(Image.open(render_p).convert("RGB")).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    fg = (np.abs(a - bg).sum(axis=2) > 30).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.GaussianBlur(fg, (3, 3), 0)
    rgba = np.dstack([np.full(fg.shape + (3,), 255, np.uint8), fg])
    Image.fromarray(rgba, "RGBA").save(out_p)
    return out_p


def _ctx_projection_cfg(ctx_dir: Path, mesh: Path,
                        prior: list[dict], args, name: str,
                        az: float, el: float) -> Path:
    """多相机投影材质的 ctx 配置（替代「图集采样」版 _ctx_cfg）。"""
    cfg = {"mesh": str(mesh), "ortho": args.ortho, "size": args.view_size,
           "views": prior,
           "camera": {"name": name, "azimuth": az, "elevation": el},
           "out_dir": str(ctx_dir)}
    p = ctx_dir / f"ctx_cfg_{name}.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _run(cmd: list[str], log: Path | None = None) -> int:
    line = " ".join(str(c) for c in cmd)
    print(f"\n$ {line}", flush=True)
    if log is None:
        return subprocess.run(cmd).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n$ " + line + "\n")
        fh.flush()
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
    for t in tail:
        print("   " + t[-160:], flush=True)
    return p.returncode


def _protect_context(gen_p: Path, ctx_p: Path, dst: Path) -> dict:
    """把 ctx 的**非品红像素原样贴回**生成图，只取 Qwen 填的品红区域。

    StableGen 默认路径是真正的 inpaint（`mask_info` 进 workflow）：
    mask 之外的像素不参与重绘，输出里原样保留。我们的
    `qwen_guided_views.py` 是整幅 img2img edit，非品红区会被反复重画，
    实测 8 轮后整体越来越暗、越来越花（v4 的褐灰脸/白腿）。
    在宿主侧做这一步保护性合成，等价于那个 inpaint mask。
    """
    import cv2

    gen = np.array(Image.open(gen_p).convert("RGB"))
    ctx = np.array(Image.open(ctx_p).convert("RGB"))
    if gen.shape != ctx.shape:
        ctx = cv2.resize(ctx, (gen.shape[1], gen.shape[0]),
                         interpolation=cv2.INTER_LANCZOS4)
    r, g, b = (ctx[:, :, i].astype(np.int16) for i in range(3))
    mag = (r > 150) & (g < 130) & (b > 150)
    mag = cv2.dilate(mag.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = np.where(mag[..., None], gen, ctx).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGB").save(dst)
    return {"edited": float(mag.mean())}


def _repair_mask(ctx_p: Path, dst: Path) -> float:
    """ctx 里的品红区 → repair mask（白=可去噪，黑=锁定）。

    StableGen `export_visibility` 的 mask 也是「未覆盖区白」；这里直接从
    投影材质渲出的 ctx 取，保证 mask 与锁定像素同源。
    """
    import cv2

    a = np.array(Image.open(ctx_p).convert("RGB")).astype(np.int16)
    r, g, b = (a[:, :, i] for i in range(3))
    mag = ((r > 150) & (g < 130) & (b > 150)).astype(np.uint8) * 255
    mag = cv2.dilate(mag, np.ones((9, 9), np.uint8))
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mag, "L").save(dst)
    return float((mag > 128).mean())


def _new_region_mask(ctx_beauty_p: Path, size: int) -> np.ndarray:
    """从 ctx render 里抽出「品红 = 尚无贴图」的区域。

    对应 StableGen `_export_visibility_impl` 的 visibility mask
    （rendering.py:1675-1685，sequential 下强制 COMPARE(1-fac, 1, eps)）：
    只标未覆盖区，已贴图区域在 inpaint 里不改。
    我们的 Qwen 引擎没有真正的 inpaint mask，就在**投影端**用这张 mask 截断，
    数学等价：整幅生成图里只有 mask 内的颜色会被投上去。
    """
    import cv2

    a = np.array(Image.open(ctx_beauty_p).convert("RGB"))
    r, g, b = (a[:, :, i].astype(np.int16) for i in range(3))
    mag = ((r > 170) & (g < 110) & (b > 170)).astype(np.uint8) * 255
    # ctx 按图集分辨率渲，降采样时用品红占比而不是最近邻，避免细缝整条丢掉
    small = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)
    m = small > 64
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return m


def _ready_with_silhouette(gen_p: Path, alpha_p: Path, dst: Path,
                           magenta_reject: bool = True,
                           new_region: np.ndarray | None = None) -> dict:
    """把生成图 + 网格剪影 alpha 合成 RGBA。

    用剪影而不是 birefnet：ctx 是同一台正交相机渲的，剪影与生成图天然逐像素
    对齐，投影覆盖因此是满的（birefnet 抠图会把 Qwen 画窄的手臂算成背景 → 洞）。

    ``new_region`` 给定时只保留该区域：顺序生成里已有贴图的部分**不再重复投影**，
    否则「图集 → ctx 渲染 → 再投回图集」每轮都会二次重采样，误差累积成
    头发花白、腿部白块（2026-11-06 实测）。
    """
    rgb = np.array(Image.open(gen_p).convert("RGB"))
    a = np.array(Image.open(alpha_p).convert("RGBA"))[:, :, 3]
    if rgb.shape[:2] != a.shape[:2]:
        raise RuntimeError(f"{gen_p} 与 {alpha_p} 尺寸不一致")
    if magenta_reject:
        r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
        mag = (r > 170) & (g < 90) & (b > 170)
        a = np.where(mag, 0, a)
    else:
        mag = np.zeros(a.shape, bool)
    if new_region is not None:
        if new_region.shape != a.shape:
            raise RuntimeError("品红区域 mask 与生成图尺寸不一致")
        a = np.where(new_region, a, 0)
    out = np.dstack([rgb, a]).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(dst)
    return {"alpha_mean": float(a.mean() / 255.0),
            "magenta_left": float(mag.mean()),
            "new_region": (float(new_region.mean())
                           if new_region is not None else 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="白模 GLB（TRELLIS.2 shape_only 产物）")
    ap.add_argument("--reference", required=True, help="角色参考图（正面）")
    ap.add_argument("--out", required=True, help="输出 GLB")
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--indices", default="", help="留空=0..cameras-1")
    ap.add_argument("--plan", help="相机方案 JSON（tools/plan_cameras.py）")
    ap.add_argument("--ortho", type=float, default=2.16,
                    help="正交视野边长，必须与 guidance/bake 一致")
    ap.add_argument("--atlas-size", type=int, default=2048)
    ap.add_argument("--view-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work", default="")
    ap.add_argument("--guide-dir", help="复用已有 guidance 输出（省一次渲染）")
    ap.add_argument("--host-python", default=str(HOST_PY),
                    help="跑 Qwen 的解释器（主项目 venv）")
    ap.add_argument("--prompt", default=SEQ_PROMPT, help="第 2..N 轮的提示词")
    ap.add_argument("--lightning", dest="no_lightning", action="store_false",
                    help="挂 8 步蒸馏 LoRA（默认不挂：StableGen QWEN 2.1 预设是 20 步）")
    ap.add_argument("--final-bake", choices=("numpy", "oneshot"), default="numpy",
                    help="numpy=逐视图烘 + Σ(color·w)/Σw 合并（已验证可用，默认）；"
                         "oneshot=单材质一次 bake（实验：本机 Cycles 5.4-alpha 有"
                         "attribute/数值限制，需 --cameras<=7）")
    ap.add_argument("--resume", action="store_true", help="跳过已完成的轮次")
    args = ap.parse_args()

    mesh = Path(args.mesh).resolve()
    ref = Path(args.reference).resolve()
    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else \
        ROOT / ".tmp" / f"seq_{mesh.stem}"
    bake_dir = work / "bake"
    ctx_dir = work / "ctx"
    gen_dir = work / "gen"
    ready_dir = work / "ready"
    for d in (bake_dir, ctx_dir, gen_dir, ready_dir, work / "guide"):
        d.mkdir(parents=True, exist_ok=True)
    progress_p = work / "progress.json"
    done: list[int] = []
    if args.resume and progress_p.is_file():
        done = json.loads(progress_p.read_text(encoding="utf-8")).get("done", [])
        print(f"续跑：已完成 {done}", flush=True)

    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        indices = list(range(args.cameras))

    # --- 相机方案 ---
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    else:
        plan_p = work / "cameras.json"
        _run([sys.executable, str(ROOT / "tools" / "plan_cameras.py"), str(mesh),
              "--out", str(plan_p), "--count", str(args.cameras),
              "--ring", "--ring-elevation", "0"], log=work / "plan.log")
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
    cam_by_index = {int(c.get("index", i)): c for i, c in enumerate(plan)}
    angles = {}
    for i in indices:
        c = cam_by_index.get(i, {})
        az = float(c.get("azimuth", 360.0 * i / args.cameras))
        el = float(c.get("elevation", 0.0))
        angles[i] = (az, el)

    # --- guidance（第 1 轮的结构图，也用来对齐尺寸）---
    guide_dir = Path(args.guide_dir).resolve() if args.guide_dir else work / "guide"
    if not (guide_dir / f"v{indices[0]:02d}_clay.png").is_file():
        rc = _run([sys.executable, str(ROOT / "tools" / "guidance.py"), str(mesh),
                   "--cameras", str(args.cameras),
                   "--indices", ",".join(str(i) for i in indices),
                   "--out-dir", str(guide_dir), "--size", str(args.view_size),
                   "--plan", str(work / "cameras.json" if not args.plan else args.plan),
                   "--ortho", str(args.ortho)], log=work / "guidance.log")
        if rc != 0:
            raise RuntimeError("guidance 渲染失败，看 .tmp 下的 guidance.log")

    # --- 逐视图循环（generator.py:1848）---
    for pos, i in enumerate(indices):
        az, el = angles[i]
        name = f"v{i:02d}"
        ready = ready_dir / f"{name}.png"
        if i in done and ready.is_file():
            print(f"\n=== [{pos + 1}/{len(indices)}] {name} 已完成，跳过 ===", flush=True)
            continue
        print(f"\n=== [{pos + 1}/{len(indices)}] {name}  az={az:.0f}°  "
              f"el={el:.0f}° ===", flush=True)
        t0 = time.time()

        if i == 0:
            # 首图：没有任何已投影内容，用参考图 + 深度结构条件生成
            gen_p = gen_dir / f"view_{i:02d}.png"
            # 白模没有 UV，拼图判据的剪影从素模渲染反推
            first_alpha = _silhouette_from_render(
                guide_dir / f"{name}_clay.png", ctx_dir / f"{name}_alpha.png")
            # **单图 + ControlNet**：两张全身图（参考图 + 素模）一起喂时，
            # Qwen 会时不时画成「左参考 / 右素模」的两格拼图；那张拼图整幅
            # 投影出来就是满身白灰斑（2026-11-06 实测）。只给参考图 + 深度
            # 条件，既不会拼图，轮廓也由 depth 钉住。
            cmd = [args.host_python, str(ROOT / "tools" / "qwen_guided_views.py"),
                   "--ref", str(ref), "--guide", str(guide_dir),
                   "--mode", "control",
                   "--cameras", str(args.cameras), "--indices", str(i),
                   "--size", str(args.view_size), "--steps", str(args.steps),
                   "--seed", str(args.seed),
                   "--plan", str(work / "cameras.json" if not args.plan else args.plan),
                   "--out-dir", str(gen_dir),
                   "--prompt", FIRST_PROMPT]
            # 首图保持挂 Lightning（8 步）：`--steps` 会被脚本 clamp；
            # 实测单图+ControlNet 这一档出的正面图最干净，别无谓换配置。
            _generate_with_guard(cmd, gen_p, first_alpha, f"{name} 首图",
                                 work / f"qwen_{name}.log",
                                 reuse=args.resume and gen_p.is_file())
            _run([sys.executable, str(ROOT / "tools" / "matte.py"),
                  str(gen_p), str(ready)], log=work / f"matte_{name}.log")
        else:
            # 1) 渲该机位的 ctx：**多相机投影材质**，不从 UV 图集采样。
            #    照 StableGen `_export_emit_image_impl`（rendering.py:940）与
            #    `_write_projection_uvw`（projection.py:360）：把 0..i-1 各相机的
            #    原始生成图按 corner 投影 UV + cos^k 权重实时求和÷权重和，
            #    未覆盖处输出品红。这样没有「图集→渲染→再投回图集」的重采样，
            #    误差不会逐轮累积（v2/v3 头发花白、腿白块的原因）。
            prior = []
            for j in indices[:pos]:
                rp = ready_dir / f"v{j:02d}.png"
                if rp.is_file():
                    jaz, jel = angles[j]
                    prior.append({"name": f"v{j:02d}", "image": str(rp),
                                  "azimuth": jaz, "elevation": jel})
            if not prior:
                raise RuntimeError(f"{name} 之前没有任何已投影视图，无法建 ctx")
            cfg_p = _ctx_projection_cfg(ctx_dir, mesh, prior, args,
                                        name, az, el)
            rc = _run([str(bake.blender_exe()), "-b", "-P",
                       str(ROOT / "tools" / "blender_context_projection.py"),
                       "--", str(cfg_p)], log=work / f"ctx_{name}.log")
            if rc != 0:
                raise RuntimeError(f"ctx 投影渲染失败（{name}）")
            print(f"  ctx：{len(prior)} 台相机实时投影（非图集采样）", flush=True)

            # 2) 剪影 alpha + repair mask（ctx 已按 view_size 渲，无需降采样）
            alpha_small = ctx_dir / f"{name}_alpha.png"
            repair_p = ctx_dir / f"{name}_repair.png"
            repair_frac = _repair_mask(ctx_dir / f"{name}_beauty.png", repair_p)
            print(f"  repair mask {repair_frac:.1%}", flush=True)

            # 3) Qwen：sampler 级 latent lock（StableGen graph_validation.py:104）
            gen_p = gen_dir / f"view_{i:02d}.png"
            cmd = [args.host_python, str(ROOT / "tools" / "qwen_guided_views.py"),
                   "--ref", str(ref), "--guide", str(ctx_dir),
                   "--mode", "beauty", "--guide-pass", "beauty",
                   "--cameras", str(args.cameras), "--indices", str(i),
                   "--size", str(args.view_size), "--steps", str(args.steps),
                   "--seed", str(args.seed), "--two-images",
                   "--plan", str(work / "cameras.json" if not args.plan else args.plan),
                   "--out-dir", str(gen_dir),
                   "--prompt", args.prompt,
                   "--latent-lock-mask", str(ctx_dir.resolve() / "{key}_repair.png")]
            if args.no_lightning:
                cmd.append("--no-lightning")
            _generate_with_guard(cmd, gen_p, alpha_small, name,
                                 work / f"qwen_{name}.log",
                                 reuse=args.resume and gen_p.is_file())

            # 4) 生成图 + 剪影 alpha → ready（整幅保留，供最终一次 bake）
            st = _ready_with_silhouette(gen_p, alpha_small, ready)
            print(f"  投影 alpha 均值 {st['alpha_mean']:.3f}，"
                  f"残留品红 {st['magenta_left']:.2%}", flush=True)

        # 逐视图烘 color/weight 到同一张 BakeUV，追加进 views/*.npy。
        # merge() 的全量 Σ(color·w)/Σw 与 project_image 的材质内归一化
        # 数学等价；首轮 unwrap 时做 BakeUV compact（rendering.py:2298）。
        if args.final_bake == "numpy":
            all_v = []
            for j in indices:
                azi, eli = angles[j]
                all_v.append({"azimuth": azi, "elevation": eli})
            got = bake.bake_views_only(
                mesh if i == 0 else (bake_dir / "baked_uv.glb"), bake_dir,
                [{"name": name, "image": ready, "azimuth": az,
                  "elevation": el, "align": "frame"}],
                size=args.atlas_size, ortho=args.ortho, exponent=7.0,
                unwrap="smart" if i == 0 else "keep",
                clear=(i == 0),
                compact=(i == 0), all_views=all_v, verbose=True)
            print(f"  已烘 {got[-1] if got else '?'}，本轮 {time.time() - t0:.1f}s",
                  flush=True)
        else:
            print(f"  {name} 完成 {time.time() - t0:.1f}s（跳过中途 bake）",
                  flush=True)

        if i not in done:
            done.append(i)
        progress_p.write_text(json.dumps({"done": sorted(done)}, indent=1),
                              encoding="utf-8")

    # --- 收尾：单材质内所有相机求和÷权重和，**一次 bake**（project_image:2248
    #     + bake_texture:2676），BakeUV 先 compact（rendering.py:2298）。
    final_views = []
    for i in indices:
        rp = ready_dir / f"v{i:02d}.png"
        if rp.is_file():
            azi, eli = angles[i]
            final_views.append({"name": f"v{i:02d}", "image": str(rp.resolve()),
                                "azimuth": azi, "elevation": eli})
    if not final_views:
        raise RuntimeError("没有任何 ready 视图")
    atlas_p = bake_dir / "atlas.png"
    fcfg = {"mesh": str(mesh), "ortho": args.ortho, "size": args.atlas_size,
            "views": final_views, "exponent": 7.0, "margin": 4,
            "unwrap": "basic", "atlas": str(atlas_p.resolve()),
            "out": str(out)}
    if args.final_bake == "oneshot":
        fcfg_p = work / "project_all.json"
        fcfg_p.write_text(json.dumps(fcfg, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        rc = _run([str(bake.blender_exe()), "-b", "-P",
                   str(ROOT / "tools" / "blender_project_all.py"), "--",
                   str(fcfg_p)], log=work / "project_all.log")
    else:
        merged = bake.merge(bake_dir, args.atlas_size, None, "weighted", 0.0)
        print(f"  最终覆盖 {merged['coverage']:.1%}，atlas {merged['atlas']}",
              flush=True)
        cfg2 = {"mesh": str(bake_dir / "baked_uv.glb"),
                "atlas": str(Path(merged["atlas"]).resolve()),
                "out": str(out), "roughness": 0.85}
        fcfg_p = work / "apply_config.json"
        fcfg_p.write_text(json.dumps(cfg2, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        rc = _run([str(bake.blender_exe()), "-b", "-P",
                   str(ROOT / "tools" / "blender_apply_atlas.py"), "--",
                   str(fcfg_p)], log=work / "apply.log")
    if rc != 0 or not out.is_file():
        raise RuntimeError("最终导出失败，看 apply.log / project_all.log")
    print(f"\n完成：{out}  {out.stat().st_size / 1e6:.1f}MB  工作目录 {work}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
