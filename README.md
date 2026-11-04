# meow3d

把 **visualbruno/ComfyUI-Trellis2 的原生纹理链**做成独立项目：一张角色图 → 带贴图的 GLB，
在项目进程内执行 ComfyUI 节点图，不启动 ComfyUI 服务。

## 为什么要独立成一个项目

主项目 `2d-game-meow` 是 **Python 3.13**，而这套节点依赖的 7 个 CUDA 编译轮子
（cumesh / nvdiffrast / nvdiffrec_render / flex_gemm / o_voxel / custom_rasterizer / natten）
在 `wheels/Windows/Torch280` 下**只有 cp311 / cp312**，cp313 一个都没有。
硬凑到 3.13 要自己编译 CUDA 扩展（本机无 nvcc），所以单独开一个 py3.12 环境，
两套互不干扰：主项目继续跑 Qwen 2D 产线，这里专跑 3D。

> 仓库 2026-02-27 之后另有 `wheels/Windows/Torch2100`（py3.13 + torch 2.10 + cu131），
> 但那条路要升 torch 大版本、cu131 更激进；选 README 主测的
> **py3.12 + torch 2.8.0 + cu128** 最稳。

## 目录

```
meow3d/
  .venv/                       CPython 3.12.13
  vendor/
    ComfyUI/                   官方 ComfyUI 源码（只借 folder_paths + comfy.cli_args）
    ComfyUI-Trellis2/          visualbruno 节点包（sparse clone，不含 wheels/）
  wheels/torch280/             6 个 cp312 编译轮子（本地留存，便于重建环境）
  models/
    microsoft/TRELLIS.2-4B/   15.12 GB
    facebook/dinov3-vitl16-pretrain-lvd1689m/   1.13 GB
  meow3d/
    runtime.py                进程内引导 + 轻量 stub + flash-attn shim
    graph.py                  UI 格式工作流执行器
    api.py                    generate(image) -> glb
  tools/
    fetch_models.sh           ModelScope 权重下载（断点续传 + 大小校验）
    generate.py               命令行
  out/                        导出的 GLB
```

## 环境与依赖

| 项 | 值 |
|---|---|
| Python | 3.12.13（`~/.workbuddy` 之外 uv 自带的那份） |
| torch / torchvision | **2.8.0+cu128** / 0.23.0+cu128 |
| CUDA | 12.8，驱动 621.01，RTX 4090 24G |
| 注意力 | 全局 `sdpa`；稀疏 `flash_attn`（由 shim 用 SDPA 实现） |
| 卷积 | `flex_gemm` |

### ⚠️ 会浪费几小时的坑

1. **Windows 的 PyPI `torch` 是 CPU 版。** `pip install torch==2.8.0` 装完是
   `2.8.0+cpu` / `torch.cuda.is_available() == False`。必须装
   `torch-2.8.0+cu128-cp312-cp312-win_amd64.whl`（3.46 GB）。
2. **镜像限速。** 阿里 `pytorch-wheels` / `pypi` 对大文件约 **200 KB/s**（3.46 GB 要 5 小时，
   pip 会看起来"卡死"），官方站走代理约 0.5 MB/s。
   **上海交大 `https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/` 实测 25 MB/s**；
   普通 pip 包走清华 `https://pypi.tuna.tsinghua.edu.cn/simple/`（50 MB/s）。
   下载好的 whl **文件名必须保持 PEP427 格式**（`torch-2.8.0+cu128-cp312-cp312-win_amd64.whl`），
   改成 `torch-cu128.whl` 会报 `Invalid wheel filename`。
3. **装编译轮子时 pip 会顺手把 torch 升到最新版。** 六个 whl 都声明依赖 torch，
   `pip install *.whl` 会拉进 `torch-2.15.0`。必须最后用
   `pip install --no-deps <本地 cu128 两个 whl>` 覆盖回来，再验证 `torch.version.cuda`。

4. **`Trellis2LoadModel` 无条件检查旧版 `TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.{json,safetensors}`**
   （和 `use_reconviagen` 无关），缺了就直连 `huggingface.co` 下，国内代理常 502。
   `fetch_models.sh` 已从 ModelScope 补这两个文件（141 MB）。
5. **输入图必须是 RGBA（4 通道）。** `Trellis2PreProcessImage.preprocess_image`
   无条件读 `np.array(img)[:, :, 3]` 当 alpha，用它算主体 bbox、裁成方形、
   再 `RGB * alpha` 合成到黑底。给 3 通道直接 `IndexError`。
   没有 alpha 的图会补一张全 255 的 alpha（等价于整图都是主体）——所以
   **最好从 2D 项目直接喂透明角色图**，白底/实景照片会被当成主体，
   生成平板或方块（主项目那条链踩过同样的坑）。

权重从 **ModelScope** 拉（约 10 MB/s），HF 走代理慢得多：

```bash
bash tools/fetch_models.sh all      # 已有且大小一致就跳过，可反复跑
```

`natten` 只用于 `TencentARC/Pixal3D`，本链路不用，因此没装。

## 用法

```bash
# 输入最好带 alpha；普通图先显式抠一下（不默认偷偷抠）
.venv/Scripts/python.exe tools/matte.py "原图.png" "out/in/角色.png"

.venv/Scripts/python.exe tools/generate.py --plan       # 只核对节点参数映射，不加载模型
.venv/Scripts/python.exe tools/generate.py "out/in/角色.png" --seed 42
.venv/Scripts/python.exe tools/generate.py "out/in/角色.png" --hq   # 官方 HighQuality 纹理参数 + 4096 UV
.venv/Scripts/python.exe tools/generate.py "out/in/角色.png" --max-size 2048  # 2K 原图不被缩到 1024

# 渲四视角预览，确认是不是真 3D（Blender）
.venv/Scripts/python.exe tools/preview_glb.py "out/meow3d/xxx_Textured_00001_.glb"
```

或在别的 Python 里：

```python
from meow3d import generate
res = generate("角色.png", seed=42)
print(res["textured"])   # 带贴图的 GLB
print(res["white"])      # 白模 GLB
```

默认跑官方 `example_workflows/MeshWithTexturing.json`：
PreProcess → Sparse(32³) → Shape(512) → ShapeCascade(1024) → DecodeLatents →
FillHoles(CuMesh) → ReconstructQuad → Simplify → FillHoles(Meshlib) → Simplify →
ToTrimesh → **MeshTexturing（texture SLat → 光栅化 UV）** → ExportGLB。
参数全部取自官方 json，不手抄；只覆盖图片、seed、导出文件名、注意力后端。

## 为什么不 import ComfyUI 整套 `nodes`

新版 ComfyUI 的 `comfy.model_management` 会拖进 `comfy_aimdo`、`comfy_kitchen`、
`comfy.model_patcher`、transformers 等一整条链。但 TRELLIS.2 节点实际只用到
`mm.current_loaded_models / get_torch_device / soft_empty_cache / free_memory /
get_total_memory` 和 `utils.ProgressBar / load_torch_file / common_upscale`。
`runtime.py` 给了等价的轻量 stub，项目因此只需要 `folder_paths` + `comfy.cli_args`
两个无重依赖的模块。同理，`Trellis2ExportMesh` 只用 `Types.File3D` 包一下路径，
真的 `comfy_api.latest` 会拉 `comfy_execution` / PyAV / 视频编解码整条链，也做了最小替身。

**路径覆盖必须在 `import folder_paths` 之前**（它在 import 时就把 `models_dir` 算死了），
做法是先改 `comfy.cli_args.args.{base,models,output,input,temp}_directory`。

## flash-attn shim

`sparse/attention/full_attn.py` 有 SDPA 兜底，但 **`windowed_attn.py` 没有**：
它无条件 `import flash_attn`，且 ATTN 不是 xformers/flash_attn 时 `attn_func_args`
根本不赋值。所以保留 `sparse_backend='flash_attn'`，由 `runtime._install_flash_attn_shim()`
注入一个 API 兼容模块，内部按 `cu_seqlens` 分段调 `torch.nn.functional.scaled_dot_product_attention`。
数学等价，代价是 Python 循环（窗口多时略慢）。**不改 vendor 一行代码**。
真装了 flash-attn 时 shim 自动放行。

## 投影贴图管线 v2：Blender 3D 表面投影 + bake（推荐）

**投影必须发生在 3D 表面，UV 贴图只是 bake 的产物。** 这是 StableGen 效果好而我们
v1 必碎的根本原因（见 `docs/StableGen复研-投影作用域.md`）。v2 用 Blender 逐三角面
正向光栅化，不改 vendor 代码、不复制 GPL 源码、**不调用 nvdiffrast / nvdiffrec**：

```bash
# 最佳配置：保留不重叠 UV + 原图正面像素直贴 + 原生贴图兜底侧背
# --size 0 = 自动跟原生贴图同边长（本例 2048）；给 1024 会明显糊
.venv/Scripts/python.exe tools/bake_projection.py 网格.glb \
    --reference out/in/knight_matte.png --out out/xxx.glb --detail 0.4

# 多相机 + 指引图（推荐）：先渲同机位结构图，再让 Qwen 按它重绘
.venv/Scripts/python.exe tools/guidance.py 网格.glb --cameras 4 --indices 0,1,2,3
"D:/code/2d-game-meow/.venv/Scripts/python.exe" tools/qwen_guided_views.py \
    --ref out/in/knight_matte.png --guide .tmp/guide_<网格名> --mode beauty \
    --cameras 4 --indices 1,2,3 --out-dir .tmp/qwen_guided
.venv/Scripts/python.exe tools/matte.py ...   # 逐个抠成 RGBA
.venv/Scripts/python.exe tools/bake_projection.py 网格.glb \
    --reference out/in/knight_matte.png \
    --views-dir .tmp/qwen_guided_matte --cameras 4 --view-indices 1,2,3 \
    --align similarity --guide-dir .tmp/guide_<网格名> --out out/xxx4.glb

# 只改合并参数、复用已烘焙的 npy（Reproject，秒级）
.venv/Scripts/python.exe tools/bake_projection.py 网格.glb --reference 任意.png \
    --out out/xxx2.glb --work-dir .tmp/bake_xxx --reproject --blend winner
```

流程：宿主侧按相机矩阵把视图**严格对齐**到模型投影 bbox（或对引导生成的图做
**相似变换**对齐）→ Blender（CYCLES）给每个相机生成 ProjUV、渲一张归一化深度图
做遮挡剔除，把「颜色」和「alpha·cos^k·可见性 权重」分别 bake 到 BakeUV →
numpy 合并 → **二进制替换原 GLB 的 baseColorTexture**（keep 模式网格一字节不改，
21MB 不变，0.03s；Blender 往返会 51MB、trimesh 会丢 metallicRoughness）。
**全量 33 条单测全过**（其中 bake/guidance 27 条，1.6 秒）。

`--align` 三选一：`bbox`（旧的 x/y 各自拉满，适合 2D 概念图当正面）、
`similarity`（等比缩放+平移，Qwen 生成的视角必须用这个）、`frame`（已同框，原样铺）。
相似变换的尺度取**面积比**初值、再用指引轮廓做两级 IoU 搜索定平移，实测
IoU 0.66~0.83。

实测（2048² 图集，17 秒）：单相机 + 原 UV + base 覆盖 45%、最干净；4 相机 +
遮挡剔除 + **指引图**覆盖 91.4%、侧背干净无污染条纹；4 相机无引导覆盖 91% 但
侧面有蓝灰脏斑。投影的定位是**还原设计**（脸是参考图的脸），原生链是
**发明细节**（锐利但脸被重绘）。
对比图 `out/knight_final_compare.png`、`out/hanfu_final_compare.png`，
头部特写 `out/heads_lit3.png`、`out/heads_guided.png`。

⚠️ **多相机的关键在「引导图」不在投影**：无引导时 Qwen 的 `view_02`（号称 90°）
实际还偏正面，投影到窄侧视 bbox 就被横向拉伸成脏斑 —— 这跟遮挡剔除是两件事。
`tools/guidance.py` 用同一台正交相机渲出深度/法线/素模/彩图，把它交给 Qwen 当
结构参考，侧背才真正对齐。详见 `docs/Blender投影bake后端.md` §7。

**ControlNet Union 在 24G 上不划算**：`models/qwen/controlnet/qwen21_union.safetensors`
（7.5GB）一挂上，显存 24.0/24.5GB，单张 768² 跑 14 分钟还在挪显存。
`--mode control` 保留但只在 40G+ 卡上值得试；默认走 `--mode beauty`
（拿模型自己的彩图当底图做指令改图，几何本来就对，也不额外吃显存）。

⚠️ **别给它两张图**：`qwen_guided_views.py` 默认只喂一张彩图。曾经把参考图一起
塞进去，Qwen 会时不时画成「左原图 / 右结果」的两格拼图（汉服两次都中招），
飘带一重叠连自动切分都救不回来。要试两张用 `--two-images`。

### 受保护精修（2026-11-05，对齐 StableGen 的 sequential 语义）

**Qwen 输出应该是「补丁」不是「整张新贴图」**。旧 `run_guided.py` 把四个机位
各自整幅重绘再全图替换，8 步 Lightning 的画风漂移被 100% 继承；而且起点是
**已带贴图的 GLB**（StableGen 是白模），整幅重绘本身就是净损失。

```bash
# 一条命令：指引图(含 flat) → Qwen 20步 → 抠图 → 保护性精修 → bake
.venv/Scripts/python.exe tools/run_guided.py out/meow3d/asuka_Textured_00001_.glb \
    --reference out/in/asuka.png --out out/meow3d/asuka_guided.glb

# 复用已生成的视图，只重跑精修 + bake（20s）
.venv/Scripts/python.exe tools/run_guided.py <网格.glb> --reference <原图.png> \
    --out out/xxx.glb --reuse-views

# 白模路线（没有可信贴图）→ 让 Qwen 当唯一颜色来源
.venv/Scripts/python.exe tools/run_guided.py <白模.glb> --reference <原图.png> \
    --out out/xxx.glb --protect 0
```

新链路多两步：`tools/guidance.py` 增加 **`flat` pass**（Emission 直出 = 贴图
原色），`tools/refine_views.py` 用它做 YUV 颜色锚定 + 局部清晰度门控的
受保护合成，输出同框 RGBA，bake 改走 `--align frame`。

⚠️ **颜色基准只能用 flat，不能用 beauty**：beauty 带灯光高光，写进贴图后
渲染会被二次打光。实测换成 flat 后，贴图相对原生的平均 |ΔRGB| 从 0.1755
降到 **0.0816（-53.5%）**、p90 从 0.3765 降到 0.2000，颜色均值基本回到原生。
`--protect 0.85` 时只改写约 22% 的前景像素。

默认采样也改成 **20 步 / 无 Lightning**（StableGen `QWEN 2.1 PRECISE`）：
单张 1024² 实跑 22.0s（8 步约 4s），与 flat 基准的色差 0.2264 → **0.1681**，
发色/战服红/五官明显更贴近原设计。要退回旧行为用 `--no-refine --lightning`。

完整源码级差距清单与优先级见 `docs/路线相同但效果较差-源码级定位与改进.md`。

### 相机布置（同日续）

旧版是 `i*360/N` 的 4 个等分**水平**机位，头顶/肩窝/腋下天然覆盖不到。
新增 `meow3d/projection/cameras.py` + `tools/plan_cameras.py`：面法线按面积
加权做球面 K-means，正面固定 index 0，俯仰 clamp 到 ±30°，可剔除朝下的面。

```bash
# 先出方案看看（Asuka 实算：0/54/107/135/171(+19°)/204(-11°)/277/359(+30°)）
.venv/Scripts/python.exe tools/plan_cameras.py <网格.glb> --count 8 \
    --out .tmp/cameras8.json

# 或让编排自动生成并跑完整链路
.venv/Scripts/python.exe tools/run_guided.py <网格.glb> \
    --reference <原图.png> --out out/xxx.glb --auto-cameras 8

# 对照旧行为
.venv/Scripts/python.exe tools/plan_cameras.py <网格.glb> --count 4 --ring \
    --out .tmp/ring4.json
```

`--plan` 会同时传给 `guidance.py` / `qwen_guided_views.py`（角度词按真实方位
与俯仰生成）/ `refine_views.py`（bbox 按真实俯仰算）/ `bake_projection.py`，
**五个环节共用同一套角度** —— 这是多视图投影最容易悄悄出错的地方。

**⚠️ 8 机位踩坑已修**：Blender 单个 mesh 最多 8 个 UV layer，原来每个视图
新建一个 `ProjUV` 会撞上限，第 8 个返回 `None`；而 `blender -b -P` 里的
Python 异常**仍以 0 退出**，宿主 `check=True` 抓不到，会静默少烘一个视图
（实测只出 7 个、覆盖 84.6%）。现在所有视图复用同一个 `ProjUV`，并在合并前
用 `bake.verify_baked_views()` 强制核对视图数 + 清掉旧 npy。

实测（20 步 + flat 精修）：8/8 视图、覆盖 **85.7%**（4 机位 80.9%，+4.8pp）、
贴图相对原生平均 |ΔRGB| **0.0703**、bake 32.1s、完整链路 5m18s。
对比图 `out/asuka_cam8_fixed_views.png`。

全量单测 **50 passed**（`tests/test_cameras.py` 7 条 +
`tests/test_bake_guard.py` 3 条回归）。

### 裁满画幅（`--fill-frame`）

原来 `ORTHO=2.8` 写死，模型归一化后最大跨度只有 2.0 → 角色在 1024² 视图里
只占 71.5%（约 732px 高）。现在画幅由 `bake.fit_ortho()` 按所有机位的
最大投影跨度自动算，并通过 `--ortho` 同步给 guidance / refine / bake：

```bash
.venv/Scripts/python.exe tools/run_guided.py <网格.glb> \
    --reference <原图.png> --out out/xxx.glb --fill-frame
```

Asuka 实测 `2.8 → 2.16`，角色 **732 → 948 px（+29.5%）**、占画幅
71.5% → 92.6%，头发/脚尖完整未裁切。画幅变了会强制重渲指引图、禁用
`--reuse-views`、禁止 Reproject 复用旧缓存（避免静默错位）。

⚠️ **反直觉结论**：最终贴图只改善约 3%（高频能量 0.0301 → 0.0310，
\|ΔRGB\| 0.0759 → 0.0747）。把 `--protect` 降到 0.5 后增益仍只有 +3.2% ——
**当前「糊」的根因不是 Qwen 视图分辨率，而是 UV 图集的 texel 密度**；
带贴图路线的底图本来就是 2048² 原生贴图，不缺分辨率。同理，20 步 vs 8 步
也只有 -4.1%：保护性合成会系统性稀释所有上游生成质量的改进。
这轮的真正价值是给**白模路线**（`--protect 0`，Qwen 是唯一颜色来源）铺路，
以及统一画幅参数、纠正归因。详见差距文档第十节。

新增 `tests/test_fill_frame.py` 6 条单测，全量单测 **56 passed**。

### v1（legacy：UV texel 反向映射）

`tools/project_texture.py` 的 `replace` 模式仍可用（正面原图直贴，1.9s），但
**`blend` / `qwen` 模式会碎玻璃**，别再用。根因记录在 `docs/投影贴图管线.md`。

**Qwen 接入方式**：权重用 `tools/link_qwen.py` 从主项目 `2d-game-meow`
**硬链接**过来（44 文件、0 复制、主项目零改动），`tools/qwen_views.py` 用主项目
venv 只读调 `Engine`（640²/8 步，3–4s/张）。Qwen 出图质量本身很好。

## 实测（2026-11-04，RTX 4090）

同一张 T-pose 汉服图，只差一个 alpha，结果天差地别：

| 输入 | 用时 | 结果 |
|---|---|---|
| 原图 RGB（不透明，浅灰背景） | **687.9 s** | 出来是**平板**：正面像、背面像，正侧视角只剩一条线。背景被当主体，TRELLIS 造了块板再把原图正反面贴上去 |
| `tools/matte.py` 抠成透明 PNG | **224.4 s** | **真正 3D**：正侧有厚度，四视角连贯，白裙/蓝纱/金绣/长发都还原，无错色斑 |
| 网上 CC0 的 2K 骑士（赛璐璐 + 面甲），`--max-size 2048` | **188.5 s** | **整体最好**：银蓝铠甲、白披风、胸前蓝纹章、金边、蓝羽饰全还原，四视角立体；初始 579 万面（简化到 50 万）。带纹理 GLB 21.0 MB |

产物（透明输入版）：
`out/meow3d/test_hanfu_matte_Textured_00001_.glb` 24.3 MB、
白模 9.0 MB，含 **2048² baseColorTexture + metallicRoughnessTexture**；
四视角预览 `out/test_hanfu_matte_Textured_00001__views.png`。

耗时大头是 `MeshTexturing` 里的 xatlas UV 展开（106 clusters / 50 万面），
可用 `--target-faces 100000` 降面加速。不透明输入之所以更慢，也是因为
它造出了更多无意义的面。

**结论**：这一版贴图质量由官方原生纹理链保证，输入图带 alpha 是硬前提。
后来为了绕开 NVIDIA 许可又做了 v2 投影 bake 后端（见上文），
两者可按需选择：原生链最省事，v2 可完全不碰 nvdiffrast。

## ⚠️ 商用许可风险（2026-11-04 补记）

之前只记了主项目底模 Qwen 的 RESEARCH LICENSE，**这里还有一条更隐蔽的**：

| 组件 | 许可 | 商用 |
|---|---|---|
| TRELLIS.2 / TRELLIS.2-4B 权重 / ComfyUI-TRELLIS2 / BiRefNet / FlexGEMM / CuMesh / O-Voxel | MIT（DINOv3 自有许可，可商用） | ✅ |
| **nvdiffrast / nvdiffrec（NVIDIA）** | **NVIDIA Source Code License** | ❌ 仅 research/evaluation，禁止任何直接或间接货币收益；只有 NVIDIA 及关联方可商用 |

我们装了 `nvdiffrast-0.4.0` + `nvdiffrec_render` 两个 cp312 轮子，
跑的又是 visualbruno 的**原生纹理链**（UV 光栅化 + PBR 烘焙正好用到它们）。
**所以商用前不止要换 Qwen 底模，还要解决 NVIDIA 这两个库。**

绕法是现成的：据 StableGen 的 *Third-party licenses*，NVIDIA 库**只在
Native (TRELLIS.2) 纹理模式**使用，**projection-based 纹理模式
（SDXL / Qwen Image Edit）完全不用**。把纹理层从原生链换成投影贴图，
整条管线就只剩宽松许可组件 —— 而且这恰好也是改善贴图/脸部的方向。

✅ **这条绕法已跑通**：`tools/bake_projection.py` 的 v2 后端不调用 NVIDIA 库，
从白模 + Smart UV + Qwen 视图也能出完整贴图（`out/knight_from_white_views.png`）。
严格声明「完全不含 nvdiffrast」还需把白模来源换成 `--workflow MeshOnly.json`
（该工作流不含 `MeshTexturing`，`--plan` 已验证）。

详见 `docs/Blender投影bake后端.md`、`docs/StableGen评估.md`
与 `docs/单图3D人脸质量-业界方案调研.md`。

## 为什么脸会「有点丑」

不是没调好参数，是**单图 3D 的条件分辨率天花板**。三层信息损失叠加：

1. **输入里脸本来就小。** 1000×1143 的全身 T-pose，脸约 140×160 px；
   `PreProcessImage` 裁到主体方形、缩到 `max_size=1024` 后只剩 ~125×145 px。
2. **DINO 条件把它压成几个格子。** 条件模型是
   `facebook/dinov3-vitl16-pretrain-lvd1689m`，**ViT-L/16**：1024 输入 →
   **64×64 个 patch token**。脸只占约 **8×9 个 token**，眼睛/鼻子/嘴各 1~2 个。
   后面所有几何和纹理都是从这 4096 个 token 里「语义级重建」，不是复制像素，
   睫毛、唇线、眼形这类细节在条件阶段就已经不存在了。
3. **几何 + 纹理各再丢一次。** 结构 32³ 体素 → shape SLat 512 → cascade 1024，
   头部只有几十个体素，五官立体感靠贴图伪造；`MeshTexturing` 用 texture SLat
   **重绘**再光栅化到 2048² UV 图集，脸只占一小块，且默认 `max_views=4`
   会把「想象出来的侧脸」一起混进正面投影。

另外展示也背了一部分锅：Blender 默认 **AgX** 色彩变换把高光压灰、皮肤发闷。
补一盏正面主光 + `view_transform='Standard'` 后肤色明显变好，
但**眼睛糊、嘴歪、头发一坨坨完全没有改善** —— 说明主因是信息丢失，不是打光。
对比图见 `out/face_compare.png`。

### 能做的（按性价比）

| 做法 | 代价 | 预期 |
|---|---|---|
| `--hq`（已实现） | 多花几分钟 | 套官方 `MeshTexturing_HighQuality.json`：25 步、cfg 7.5、`max_views=1`（不让侧脸污染正面）、`dino_lock=0.35`（后期采样贴回 DINO 条件）、UV 图集 4096。改善清晰度与还原度，**但救不回几何** |
| **脸部高清贴图替换** | 要写投影/UV 定位 | 把原图那张高清脸按 UV 贴回头部图块、边缘羽化。像素级还原，最确定 |
| 多视角工作流 | 需要额外侧/背图 | 官方 `MeshWithTexturing_MultiView.json` / `Projection_*_Qwen_*Views.json`，改善侧面与背面 |
| 换构图 | 丢全身 | 半身/胸像让脸在 DINO token 里占比翻倍 |

结论：这套链路擅长还原**整体造型、衣服配色、飘带层次**（这次都对了），
不擅长还原**五官**。真要脸精致，走「贴图替换」或多视角，别指望单图再加步数。

## 网络测试素材

`out/in/knight_tpose.png` —— **Servant Knight Concept T-Pose**，1536×2048 PNG，
**CC0**（无需署名、可商用），作者 Xiomara Gonzalez，Summer Engine，2026-10-13：
<https://www.summerengine.com/asset-store/servant-knight-concept-t-pose-ac0708bc>

选它的理由：许可干净、纯白底好抠、标准正面 T-pose 且头到脚完整、银蓝铠甲 +
纹章 + 金边 + 披风的细节足够密、无文字水印。抠图版 `out/in/knight_matte.png`
（BiRefNet，白色披风保留完好、无白边）。

⚠️ **高清输入并不等于更高清的条件**：`PreProcessImage` 默认 `max_size=1024`
会先把 2K 图缩掉（用 `--max-size 2048` 可保住）；但 DINO 条件固定
`cond_resolution = min(resolution, 1024)`、`image_size=1024`，仍会 resize 成
**64×64 patch token**。所以提高输入分辨率主要惠及纹理投影/烘焙，
**不会增加脸在条件里占的 8×9 个格子** —— 这正是「脸丑」那一节的结论。

## 参考上游

- `microsoft/TRELLIS.2` —— 3D 生成模型
- `visualbruno/ComfyUI-Trellis2` —— ComfyUI 封装、编译轮子、原生纹理链、示例工作流
- `AHEKOT/ComfyUI_VNCCS` —— 主项目 2D 产线的参考（与本项目无关，仅说明来源关系）
