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
参数全部取自官方 json，不手抄；在此之上只覆盖图片、seed、导出文件名、注意力后端，
以及下一节按 StableGen 默认值对齐的 5 项。

## 前半段与 StableGen 逐节点核对（2026-11-05）

复刻的链是「**图片 → 3D 模型 → 骨骼绑定**」，核对顺序必须先做前半段，前半段没
对齐不进绑定。StableGen 的 image→mesh 用 `PozzettiAndrea/ComfyUI-TRELLIS2`
（pin `9b878516`，`installer.py:613,618`）的 9 节点模板
`util/workflow_templates.py:1642-1762`；本项目用 `visualbruno/ComfyUI-Trellis2`
的 24 节点 `MeshWithTexturing.json`。**两套节点包节点名/参数名不同，无法逐节点
1:1 复制**，因此按语义逐项对齐默认值（用户选定的「进程内、做法对齐」路线）。

| 环节 | StableGen（源码 : 行） | 默认 | meow3d 原值 | 判定 |
|---|---|---|---|---|
| 自动抠背景 | `workflow_templates.py:1665-1674` `Trellis2RemoveBackground`；`mesh_gen/workflows.py:1279` | **开（BiRefNet）** | `remove_background=false`，靠手动 `matte.py` | ❌ 头号 |
| 背景合成 | `:1680 background_color="black"` | black | `nodes.py:3215` 黑底 | ✅ |
| 模型/分辨率 | `:1654-1664 1024_cascade/auto/auto`；`properties.py:1592,1601,1611` | 1024_cascade / bf16 | `LoadModel` + 512→1024 cascade | ✅ 等价 |
| 稀疏结构 | `:1687-1702 ss=12/g=7.5, shape=12/g=7.5, max_tokens=49152` | 12 / 7.5 / 49152 | `[214] 12 / 7.5 / 32³` | ✅ |
| 形状 / 级联 | 同上 + `:1703-1717 tex_steps=12` | 12 | `[215]/[216] 12 / 7.5` | ✅ |
| 贴图图集 | `:1738-1749 RasterizePBR(texture_size=4096)`；`properties.py:1655` | **4096** | `[260] 2048` | ❌ |
| 纹理引导 | `mesh_gen/workflows.py:1340` ← `properties.py:1642` | **3.0** | `[261] 5` | ❌ |
| 纹理步数 | `properties.py:1646` | 12 | `[261] 12` | ✅ |
| 网格后处理 | `:1718-1737 ProcessMesh(target_face_count=100000, remove_inner_faces=false, remesh="on", weld=true, chart_cone_angle=90)` | **10 万面 / false** | `[161] true` + `[218]/[257] 50 万面` | ❌ |
| 导出 | `:1750-1760 ExportTrimesh(glb)` | 1 个 GLB | 白模 + 贴图 2 个 | ➕ 无质量影响 |
| seed | `:1691,1709` | 0 | `--seed 42` | ➖ 可控 |

**关键事实**：`Trellis2MeshTexturing` 节点自带默认就是
`texture_size=4096 / texture_guidance_strength=3.00`（`nodes.py:2828,2832`），
数值上等于 StableGen 默认。**但这条推理随后被证伪了：数值相同 ≠ 语义相同**，
见下一节。

### ⚠️ 否定结论：抄参数不是「一模一样」，而且会把模型改坏（2026-11-05）

曾把 5 项抄成 StableGen 默认值（自动抠图 / 4096 / guidance 3.0 / 10 万面 /
`remove_inner_faces=false`）。**默认路径已全部回滚**，`--stablegen-align` 仅作
实验开关保留，**不要启用**。

**严格 A/B**（同 `knight_matte.png`、seed 42、`--max-size 2048`，只差这 5 项）：

| | 官方原值（对照） | 抄参数（实验） |
|---|---|---|
| 产物 | `knight_2k_Textured_00001_.glb` | `knight_ab_aligned_Textured_00001_.glb` |
| 贴图 | 2048² | 4096² |
| 三角形 | **460365** | **25641** |
| 四视角 | 完整角色 | **只剩几条细线，模型几乎没了** |

实证 `out/knight_ab_aligned_Textured_00001__views.png`。

根因是**参数语义错配**（源码级）：

- StableGen `target_face_count=100000` 在 `Trellis2ProcessMesh`
  （`workflow_templates.py:1718-1737`）里是**最终网格**，随后直接
  `RasterizePBR` → 导出；meow3d `target_face_num` 在 `SimplifyMesh` 里是
  **中间网格**，`Trellis2MeshTexturing` 还会做 xatlas 聚类（日志
  `Get 4906 clusters`）再合并一次 → **连砍两次面**，几何塌成薄片。
- `remove_inner_faces` 在 `ReconstructMeshWithQuad`（quad 重建阶段）与
  `Trellis2ProcessMesh`（QDC 重网格阶段）语义不同，照抄会让内部面参与聚类。
- 两边**纹理算法本身就不同**：StableGen 是 `ShapeToTexturedMesh` 出
  `voxelgrid` → `RasterizePBR` 烘到网格；meow3d 是 `MeshTexturing` 做 texture
  SLat 扩散 + xatlas UV 展开。改数值改不了算法。

另外上一版「端到端 161s、效果更好」的结论**作废**：那次用的是默认
`max-size 1024`，对照组旧产物是 `--max-size 2048`，且 5 个变量同时变，
根本没控制变量，拿它比画质不成立。自动抠图本身已验证无害
（`knight_tpose.png` 经 `birefnet-general-lite` 抠出的 alpha 与
`knight_matte.png` 平均绝对差 **0.0000**），但它救不了参数错配。

**结论：真正的「一模一样」只能换节点包** —— 用 StableGen 指定的
`PozzettiAndrea/ComfyUI-TRELLIS2`（pin `9b878516`）的 9 节点
`prompt_text_trellis2` 模板在进程内执行；在 `visualbruno` 上抄数值永远到不了。
这一条尚未实施。

**当前默认**：`align_stablegen=False` / `bg_removal='skip'`，即官方
`MeshWithTexturing.json` 原值，已回到未劣化状态。

```bash
# 默认（官方原值）
.venv/Scripts/python.exe tools/generate.py "out/in/knight_matte.png" --seed 42 --max-size 2048
# 不透明原图：显式开自动抠图（已验证与手工 matte.py 等价）
.venv/Scripts/python.exe tools/generate.py "out/in/knight_tpose.png" --bg-removal auto
# 只看真实执行参数（不加载模型）
.venv/Scripts/python.exe tools/generate.py --plan
```

**没做的事**：没换节点包、没动投影/贴图链、**没进绑定阶段**。`--hq` 仍按官方
HighQuality 工作流覆盖纹理参数，与 StableGen 默认是两套配置。

## StableGen 9 节点链：真·一模一样（进程内，2026-11-05）

上一节证明了「把数值抄到另一套节点包」行不通。真复刻 = 用 StableGen 指定的
**同一套节点实现** `PozzettiAndrea/ComfyUI-TRELLIS2` pin `9b878516`，执行同一份
9 节点 **API 格式 prompt** `prompt_text_trellis2`。本项目在**当前进程**里跑它，
不启动 ComfyUI 服务。

```bash
# 1) 一次性：把现有 15GB 权重按节点包目录约定摆好（硬链接，0 复制）
.venv/Scripts/python.exe tools/link_sg_models.py
# 2) 只看最终 prompt（不加载模型）
.venv/Scripts/python.exe tools/generate_sg.py --plan
# 3) 跑
.venv/Scripts/python.exe tools/generate_sg.py "out/in/knight_matte.png" --seed 42
```

| 环节 | 做法（源码位置） |
|---|---|
| 9 节点 prompt | `meow3d/sg_prompt.py:PROMPT_TEXT_TRELLIS2`，参数逐项照 `workflow_templates.py:1642-1762` + `mesh_gen/workflows.py:1258-1360` 的运行时覆盖（tex_guidance **3.0** / texture_size **4096** / 面数 **10 万**） |
| 节点实现 | `vendor/_sg/...` 原样代码，由 `meow3d/sg_runtime.py:bootstrap()` 装载；9 个 `execute()` 全是 classmethod，可直接调用 |
| io 描述层 | `meow3d/sg_io.py` 替身（真 `comfy_api.latest` 会拉整套 ComfyUI 节点链） |
| ComfyUI 运行时 | `sg_runtime.py` 轻量 mm / utils / ModelPatcher 替身；`comfy_kitchen` 用真包 |
| 编译轮子 | `cumesh_vb→cumesh`、`flex_gemm_ap→flex_gemm`、`o_voxel_vb_ap→o_voxel` 模块别名（API 已逐项核对一致，不重编译） |
| 权重 | `tools/link_sg_models.py` 把 `models/microsoft/TRELLIS.2-4B/ckpts/` 硬链接到 `models/trellis2/ckpts/`（节点包 `stages.py:169` 只认这个目录） |
| drtk | PyPI 同名 100.0.0 是占位包，真包走 pixi；用 `meow3d/sg_drtk.py` 等价实现 `rasterize` / `render` |

**三处明确偏差**（刻意取舍，不是猜漏）：
1. 背景去除默认走 StableGen 的 **skip 分支**（`mesh_gen/workflows.py:1267-1277`）：
   删掉 `Trellis2RemoveBackground`，输入需自带 alpha。`--bg-removal auto` 可切回
   默认分支，但要先准备 HF `ZhengPeng7/BiRefNet` 权重。已验证宿主侧 BiRefNet
   抠图与该节点 alpha 平均绝对差 0.0000。
2. `LoadImage` 自己实现（ComfyUI 内置版依赖 PyAV / VideoFromFile）；mask 给
   **前景=1**，与 `Trellis2RemoveBackground.execute` 的输出一致（注意 ComfyUI
   内置 LoadImage 给的是 `1-alpha`，不能混用）。
3. `drtk` 用等价替身。

⚠️ **许可提示**：`sg_drtk.py` 借本机 `nvdiffrast` 做 UV 空间光栅化，属 NVIDIA
Source Code License（仅 research/evaluation）。本项目原本的 v2 投影后端刻意不用它；
商用前需把 `_rasterize_uv` 换成纯 torch 实现。

**依赖**：新装 `comfy-kitchen 0.2.41` / `comfy-sparse-attn 0.1.3`（均 `--no-deps`，
torch 保持 2.8.0+cu128）；复用本机 cumesh / flex_gemm / o_voxel / nvdiffrast 与
TRELLIS.2-4B 权重。新增 `tests/test_sg_prompt.py` 6 条 + `tests/test_sg_drtk.py`
2 条回归单测。

### 实测（2026-11-05，RTX 4090）

```
.venv/Scripts/python.exe tools/generate_sg.py "out/in/knight_matte.png" --seed 42
→ 74.9s
→ out/knight_sg2_bfe5f11b_20261105_220839.glb  24MB
   1 个 PBR 材质 / 2 张 4096² 贴图 / 99828 tris
→ 四视角 out/knight_sg2_bfe5f11b_20261105_220839_views.png
```

银白铠甲、蓝金纹章、白披风、蓝羽饰、金边全部清晰，正侧背连贯，无碎裂斑块。

⚠️ **drtk 替身的关键坑（别再踩）**：nvdiffrast 输出数组 `y=0` 对应画面底部，
而节点包 `nodes_unwrap.py:1046-1076` 把 `(0,0)` 当左上、y 向下使用。少了
`ids.flip(-2)`，`rast_face_ids` 上下镜像 → `valid_pos` 采样到错误 3D 位置 →
贴图变成碎裂迷彩斑（第一版 `out/knight_sg_641a444a_20261105_220401_views.png`）。
最小三角形 + 重心插值回归测试见 `tests/test_sg_drtk.py`（插值误差 0.0）。

全量单测 **80 passed**（当轮；后续补 Qwen 2.1 链后为 83 passed）。

### 画质对比与归因（2026-11-05）

四列对比图 `out/knight_4way_compare.png`（原图 / 旧链 / 新链默认 / 新链白底），
四视角 `out/knight_sg_vs_old.png`，贴图 `out/knight_basecolor_compare.png`。

结论：**「一模一样」与「这张白底角色图更好看」在当前是冲突的**——视觉上旧链
（visualbruno 2048 / 46 万面）更白净贴原图；新链金边/纹章更锐利、结构更干净，
但整体偏灰暗、脸是橙黄色平板。

| 版本 | baseColor 平均亮度 | p90 | 饱和度 | 视觉 |
|---|---:|---:|---:|---|
| 旧链 visualbruno | **101.8** | 187.0 | **0.443** | 最白净、蓝金饱和 |
| 新链默认（black/g3/10万面） | 82.1 | 148.0 | 0.394 | 偏灰暗、脸橙黄 |
| 新链 50 万面 | 82.4 | 149.0 | 0.395 | 与 10 万面几乎逐像素相同 |
| 新链 tex_guidance 5.0 | 84.7 | 154.0 | 0.414 | 轻微改善，脸未变 |
| 新链 background_color white | **110.5** | 182.0 | 0.333 | 提亮但褪色，且**手臂/胸口出现黑色斑块** |

**排除项（别再试）**：① 面数不是原因（10 万 → 50 万贴图亮度只 +0.3，
四视角几乎一样）；② PBR 金属度不是原因（三版 metallicFactor=1、
MR 贴图 B 通道≈0、G 通道≈254，材质参数完全相同，差异在 baseColor 内容本身）。

**真正的差异在源码里**：
1. 输入预处理：旧链 `preprocess_image` 是 `RGB × alpha` 预乘黑底 +
   按 bbox **无留边**裁方；新链 `smart_crop_square(margin_ratio=0.1,
   background_color=black)`（`stages.py:828`）带 10% 留边、黑底面积更大 →
   条件图里主体占比不同。
2. 纹理算法：旧链 `Trellis2MeshTexturing`（texture SLat 扩散 + xatlas 直接烘）；
   新链 `ShapeToTexturedMesh → voxelgrid → RasterizePBR`（从体素属性采样到 UV）→
   白色/肤色区域更容易串色（橙黄脸、深蓝溢出）。
3. `tex_guidance` 默认 3.0（StableGen）vs 旧链 5；A/B 只带来 +2.6 亮度。
4. 脸部糊是单图 3D 的条件分辨率天花板（见「为什么脸会有点丑」），两条链都有，
   新链更明显。

**取值建议**：要严格复刻就保持默认 `black / 3.0 / 10万面`，接受偏暗（不是 bug）；
要更贴原图，旧链或后续接 v2 投影 bake 把原图脸部/白色区贴回去更直接。
`--background-color white` 虽然亮度数字最高，但会引入黑斑，**不推荐**。
`--tex-guidance 5.0` 是无副作用的轻微改善。

#### 换一张图复测：Asuka（2026-11-05，补漏）

用 `out/in/asuka_ref.png`（浅灰底红衣橙发 T-pose）抠图后跑同一条 9 节点链
（92.2s，37MB GLB）。`out/asuka_lit_vs_flat.png` 是决定性证据：

- 带灯光预览：正面/背面**整片死白**（头发白、战服粉白斑驳）；
- `--flat`（Base Color 直连 Emission）：**完全正常**，红衣/橙发/绿领/黑条纹/脸都对。

原因不是贴图错，是 **PBR roughness 太低**：Asuka 的 metallicRoughness 贴图
G 通道只有 **39/255**（红色漆皮被判成高光滑表面），骑士是 254；
`blender_preview.py` 的 4 盏 AREA 灯（总能量 1540）打出过曝高光。
→ **修正上一节的过度推广**：「PBR 不是原因」只对骑士那张成立，不能推到所有输入；
**判断贴图颜色必须看 FLAT 版**，灯光版只看立体感。

```bash
.venv/Scripts/python.exe tools/preview_glb.py <模型.glb> --flat   # 贴图本色
```

同等 FLAT 条件下 `out/asuka_flat_compare.png`：新链橙发比旧链金发更贴原图、
脸有五官；代价是红衣偏暗、身形偏细。两版各有优劣，**新链并不比旧链差**。

**逐项补漏核对**（StableGen 场景默认值 vs 实际传值）：
- ⚠️ **已修**：`precision` 模板写 `auto`，但 `mesh_gen/workflows.py:1263` 会用
  `scene.trellis2_precision` 覆盖，`properties.py:1601-1610` 默认是 **`bf16`**；
  `sg_prompt.py` 已改 bf16（单测锁定）。
- 已核对一致：`resolution=1024_cascade`、`attn_backend=auto`、`seed=0`、
  `ss/shape_guidance=7.5`、`ss/shape_steps=12`、`max_tokens=49152`、
  `tex_guidance=3.0`、`tex_steps=12`、`texture_size=4096`、`decimation=100000`、
  `post_processing=True`、`decimate_method=server`、`remesh_method=qdc`、
  `background_color=black`、`skip_texture=False`、`autorig=False`、`solidify=False`。
- 9 节点里只有 3 项走 schema 默认值（`ss/shape_guidance_rescale=0.05`、
  `tex_guidance_rescale=0.2`），StableGen 模板同样没传，属正常。
- 已知偏差：`bg_removal` 默认 `auto` 用 `briaai/RMBG-2.0`，我们用 `skip` +
  `birefnet-general-lite`。但 `Trellis2RemoveBackground.execute`
  (`nodes_inference.py:258-313`) 返回的是**原图 + alpha mask**，与我们的
  `LoadImage` 数据流一致，只有抠图模型不同。`_rasterize_alpha`
  (`mesh_gen/workflows.py:102-141`) 只作用于 skip 分支，默认 auto 不走。
- 不影响的漏项：StableGen 导入 Blender 后还会做 `import_scale=2.0` /
  flat shading / 清 custom normals / 自动打光，这些改的是场景展示，不改 GLB 内容。

## Qwen Image 2.1 在 StableGen 里的位置（v0.5.0 校正，2026-11-05）

⚠️ **两次纠正的记录**（避免再拿旧结论当事实）：
1. 先说「Qwen 2.1 是招牌链的一半」——没查预设，说大了；
2. 随后用本地 `D:/code/gen/others/StableGen-main` 的预设表反驳自己，
   得出「MESH 预设没有 Qwen 2.1」——**这一步也错了**，因为那份是**旧快照**。
   经代理拉取 `sakalond/StableGen` 最新代码后确认是 **v0.5.0**
   （`stablegen/__init__.py` → `"version": (0, 5, 0)`）。

### v0.5.0 的真实预设（`stablegen/ui/presets.py:260-280`）

`_TRELLIS_QWEN_21` 循环把四个 `QWEN * (MESH + TEXTURE)` **原地替换**为
`QWEN 2.1 * (MESH + TEXTURE)`，源码注释原文：

> "TRELLIS.2 meshes textured with Qwen-Image 2.1, which replaced Qwen
> Image Edit's: each keeps its TRELLIS.2 mesh, camera and texturing setup
> and takes Qwen 2.1's tested sampling."

| MESH + TEXTURE 预设 | `trellis2_texture_mode` |
|---|---|
| DEFAULT / CHARACTERS / ARCHITECTURE | `sdxl` |
| **QWEN 2.1 PRECISE / SAFE / ALT / VORONOI** | **`qwen_image_21`** |
| TRELLIS.2 (MESH ONLY) | `none` |

四个 Qwen 2.1 MESH 预设的固定参数：`steps=20`、`cfg=1.0`、`dpmpp_sde`、
`sgm_uniform`、`weight_exponent=7.0`（Voronoi 用自己的）、
`qwen_image_21_native_rgba=False`、`qwen_guidance_background_color` 灰 0.5。

⚠️ **`native_rgba=False` 是 v0.5.0 特意改的**（`core/properties.py:429-437`）：
"Off by default: the model also cuts away surfaces it should paint, such as
ground and walls, which leaves them for worse-angled views. Transparent
generated pixels are never projected"。旧版默认是 True，上一轮我按旧版理解。

### v0.5.0 把 TRELLIS.2 整个换成了 ComfyUI 内置节点

新文件 `mesh_gen/trellis2_graph.py`（185 行）直接构建原生节点图，
`mesh_gen/workflows.py:1374` 调用 `trellis2_graph.build_trellis2_prompt`，
**旧 `prompt_text_trellis2` 模板已不再使用**。节点序列：

`LoadImage` → `UNETLoader(trellis_2_bf16)` → `VAELoader(shape_vae)` →
`CLIPVisionLoader(dino_v3_vit_l)` → `LoadBackgroundRemovalModel` +
`RemoveBackground` → `ImageCropToMask` → `Trellis2Conditioning` →
`ModelSamplingSD3` + `RescaleCFG` + `CFGOverride` → `KSampler`(结构) →
`VaeDecodeStructureTrellis2` → `Trellis2ShapeStage` + `KSampler`(形状) →
`Trellis2UpsampleStage` + `KSampler`(高分辨率) → `VaeDecodeShapeTrellis` →
`RemeshMesh` / `DecimateMesh` →（纹理）`Trellis2TextureStage` + `KSampler`
→ `VaeDecodeTextureTrellis` → `UnwrapMesh` → `BakeTextureFromVoxel` →
`ApplyTextureToMesh` → `SaveGLB`

**本机 ComfyUI 是 0.43.0**（`vendor/ComfyUI/comfyui_version.py`），上述节点
全部已存在（`comfy_extras/nodes_trellis2.py`、`nodes_mesh_postprocess.py`、
`nodes_bg_removal.py`、`nodes_save_3d.py`、`nodes_images.py`）。

### 这对我们意味着什么

- 我们复刻的 9 节点链（`LoadTrellis2Models` / `Trellis2ImageToShape` /
  `Trellis2ShapeToTexturedMesh` / `Trellis2ProcessMesh` / `Trellis2RasterizePBR`
  …）属于 **v0.3.1 的自定义节点时代**，v0.5.0 已整体废弃。
- 为此写的 `meow3d/sg_drtk.py`（nvdiffrast 替身）、`sg_runtime.py` 的
  io/mm/ops 替身、`tools/link_sg_models.py` 的 26 个硬链接，在新链上
  **都不再需要**——官方正是把这些依赖去掉了。
- 代价：新链要 5 个单文件权重，合计约 **13.1GB**
  （`trellis_2_bf16` 9.63 + `shape_vae` 1.02 + `texture_vae` 0.88 +
  `dino_v3_vit_l` 1.13 + `birefnet` 0.41），源 `Comfy-Org/TRELLIS.2` 与
  `Comfy-Org/BiRefNet`（`stablegen/setup/installer.py:378-410`）。
  本机只有 TRELLIS.2-4B 的原始 ckpts 分件，文件名与格式都不匹配。

### v0.4.0 / v0.5.0 其余关键更新

- **v0.4.0**：Qwen Image 2.1 首次引入（Precise / Precise+ / Safe /
  Context + Reference / Local Edit / Refine 六个预设）、原生支持 2K
  （约 4MP/视图）、RGBA 视图透明像素不参与投影、**Masked Edit**
  （画选区只重生成该区域）、Auto-rig（SkinTokens）+ FBX/BVH 动作重定向、
  实验架构 Qwen 2.1 Turbo（8 步，约 3 倍速）/ MOSS Image / FIBO 2.0、
  Marigold V2、新面板。
- **v0.5.0**：Blender 内安装/运行 ComfyUI；TRELLIS.2 原生节点化（不再需要
  自定义节点、独立 Python 环境、编译扩展、nvdiffrast）；**LATO.2 低模重网格**
  （200–5000 顶点）；Make Solid 改 flood fill + Close Bottom Openings；
  Post-Process Active Mesh；**Object-Relative Projection**（物体移动贴图不裂，
  Blender 5.1+）；**Softer seams**；Confidence-Gated Context；Diagnostics
  16.7s → 0.03s。

> 下面「本机条件 / 本次接法 / 实测」三段基于**旧快照 v0.3.1 链**，
> 仅作基线保留；v0.5.0 迁移后这部分会被替换。

**本机其实早就具备条件，只是没串起来**
- `models/qwen/checkpoints/qwen-image-2.1/`：diffusers 全量权重约 58GB，
  由 `tools/link_qwen.py` 从主项目 `2d-game-meow` 硬链接（44 文件、0 复制）
- `tools/qwen_views.py` / `qwen_guided_views.py`：调主项目
  `workflows.engine.Engine`（py3.13 / torch2.6 / diffusers0.41，已验证可 import）
- `tools/run_guided.py`：一条命令「指引图 → Qwen → 抠图 → 保护性精修 → bake」，
  参数已按 `QWEN 2.1 PRECISE` 对齐（20 步 / CFG1 / 无 Lightning / 8 机位 /
  cos^k），详见上文「投影贴图管线 v2」
- `vendor/ComfyUI/comfy_extras/nodes_qwen.py` 里也有 `TextEncodeQwenImage21`
  与 `QwenImage21Cache`，正是 installer 要求的 `packages` 节点

**本次补上的接法**（默认行为不变，`generate_sg.py` 仍是 native）
⚠️ 这是 StableGen **枚举允许但无预设**的组合，只作能力验证，不代表官方路径。
1. `meow3d/sg_prompt.py` 新增 `PROMPT_TEXT_TRELLIS2_SHAPE_ONLY`，照
   `util/workflow_templates.py:1764-1852` 原文，并按
   `mesh_gen/workflows.py:1327-1330` 置 `remesh=True`；用
   `build_prompt(skip_texture=True)` 切换，`tools/generate_sg.py --shape-only` 暴露。
2. `tools/generate_sg_qwen21.py` 编排：白模 → `run_guided.py --white-mesh
   --auto-cameras 8 --steps 20 --exponent 7.0` → 成品 GLB。
   `--plan` 打印 StableGen 预设值与本地传值对照，新增 3 条单测锁定模板。

```bash
.venv/Scripts/python.exe tools/generate_sg_qwen21.py --plan
.venv/Scripts/python.exe tools/generate_sg_qwen21.py \
    out/in/asuka_ref_matte.png --reference out/in/asuka_ref.png \
    --seed 42 --out out/asuka_qwen21.glb
```

⚠️ **偏差，别当成逐节点复刻**：StableGen 用 ComfyUI 节点图 + int8 单文件
权重；这里用主项目 diffusers Engine + 本地 diffusers 权重，属于**做法对齐**。
StableGen sequential 的「品红 context render + 保留非品红区域」由
`refine_views.py` 的 flat 基准保护性合成近似；Qwen 输出经 `matte.py` 抠图，
不是 Qwen 2.1 原生 RGBA。要真正逐节点一样，需按 installer package 13 另下
约 14GB ComfyUI 格式权重，并把 `TextEncodeQwenImage21` 接进 `sg_runtime`。

⚠️ 更关键的一条：**本地没有 Qwen Image Edit 2509 权重**（只有
`qwen-image-2.1`），所以 StableGen MESH 预设真正使用的
`qwen_image_edit` 官方路径，目前在本项目也跑不了。要复刻那条官方路径，
得先补 2509 权重。

### 实测（Asuka，seed 42，RTX 4090，2026-11-05）

| | native 9 节点链 | Qwen 2.1 投影链 | + `--fill-frame` |
|---|---:|---:|---:|
| mesh 阶段 | 92.2s（直接带贴图） | 28.3s 白模 | 复用白模 |
| 整链耗时 | 92.2s | **374.2s** | 319.8s |
| 贴图覆盖 | 原生 UV 全覆盖 | **33.9%** | 34.4% |
| UV 岛占用 | — | 13.6% | 13.6% |
| 抠图视图不透明占比 | — | 6.1% | 7.9% |
| 成品 | 37MB | 6.0MB | 6.0MB |

⚠️ **结论：链接上了，但目前不推荐当默认**。四视角
（`out/asuka_qwen21_flat_views.png`）全是破碎迷彩斑，正/背面尤差、脸糊；
native 链干净完整。Qwen 原视图本身**没问题**
（`out/asuka_qwen21_views_raw.png`：完整 T-pose、红衣橙发都对），
坏在投影环节：smart unwrap 的 UV 图集只占 13.6%，8 视图每个仅覆盖
约 6~7% texel，合并只有 34%；空洞靠 diffuse 兜底打成斑驳，再叠加
侧视 v02 被 Qwen 画成收臂站姿（上文「白模路线」已记录的老毛病）
与 T-pose 网格错位。`--fill-frame` 把视图主体从 6.1% 提到 7.9%，
覆盖率只 +0.5pp、画质几乎不变 → **画幅不是瓶颈**。

StableGen 能把这条链跑好，靠的是 Blender 里逐面 3D 表面投影
（BVH raycast × cos^k）再 bake 到 `BakeUV`；本项目 v2 投影是
UV texel 反向映射 + smart unwrap，覆盖率天花板不同。要到 StableGen
的覆盖水平得换投影实现，不是继续调 Qwen 参数。

产物 `out/asuka_qwen21.glb` / `out/asuka_qwen21_fill.glb`（各 6.0MB），
对比图 `out/asuka_qwen21_compare.png`（五列）与
`out/asuka_qwen21_ab.png`（A/B）。全量单测 **83 passed**。

## v0.5.0 原生节点链已跑通（2026-11-06）

**结论**：v0.5.0 的 ComfyUI 内置节点链在本机 24GB / torch 2.8+cu128 上可用，
且比 v0.3.1 自定义节点链**快一倍**：

| | v0.3.1 自定义 9 节点 | v0.5.0 原生（512） | v0.5.0 原生（1024_cascade） |
|---|---:|---:|---:|
| 耗时 | 92.2s | 39.2s | **44.4s** |
| 产物 | 37MB | 13MB | 14MB |
| 面数 | 99828 | 99811 | — |
| 贴图 | 4096²×2 | 4096²×2 | 4096²×2 |
| 画质 | 红色发白/斑驳 | 红色饱和 | 红色干净、细节更好 |

对比图 `out/asuka_native_compare.png`（原图 / 旧链 / 512 / 1024）。

### 之前「12 分钟 + OOM」是我漏了一层，不是 v0.5.0 的问题

第一次直调 `cls.execute()` 没包 `torch.inference_mode()`：ComfyUI 的
`execution.py:757` 用它包住整条链，少了这层，每个中间张量都带 `grad_fn`，
显存从 ~2GB 飙到 24GB 打满、GPU 100% 却只有 96W（显存颠簸），512 白模
12 分钟出不来。补上后同一张图 **44.4s**。

另外三处「只有直调才需要」的适配（前端/服务端本来会替你补）：

1. **DynamicCombo 要 dict**：`sign_mode={"sign_mode":"udf"}`、
   `placement_mode={"placement_mode":"midpoint"}`；给字符串会报
   `'str' object has no attribute 'get'`。
2. **`cls.hidden` 注入**：`RemeshMesh` 读 `cls.hidden.unique_id` 发进度，
   服务端由 `execution.py:198-207` 注入，直调要给 `HiddenHolder`。
3. **`server.PromptServer.instance`**：`nodes_mesh_postprocess` 会调
   `PromptServer.instance.send_progress_text(...)`，给 no-op 实例即可。

### 用法

```bash
bash tools/fetch_models.sh trellis2native    # 5 个单文件 13.1GB，魔塔 10-29MB/s
.venv/Scripts/python.exe tools/generate_native.py --plan
.venv/Scripts/python.exe tools/generate_native.py out/in/asuka_ref_matte.png \
    --seed 42 --prefix asuka_native             # 默认 1024_cascade，约 45s
```

新增 `meow3d/sg_native.py`（图构建 + 执行）、`tools/generate_native.py`、
`tests/test_sg_native.py`（9 条）。依赖只补了 `av` 与 `torchsde`。
不再需要 comfy_env / comfy_sparse_attn / cumesh 别名 / drtk 替身。

⚠️ 待查：四视角里**脸部偏白块**（三版都有，v0.5.0 更明显），
疑似头部 UV 覆盖不足，下一步单独排查。

## 白模 + Qwen 2.1 重贴（v0.5.0 的成品路径，2026-11-06）

v0.5.0 的 MESH + TEXTURE 预设**没有一个用 native**：`qwen_image_21` 4 个、
`sdxl` 3 个、`none` 1 个。所以官方成品 = 白模 + 扩散重贴，本项目按这条接：

```bash
.venv/Scripts/python.exe tools/generate_native_qwen21.py \
    out/in/asuka_ref_matte.png --reference out/in/asuka_ref.png \
    --seed 42 --out out/asuka_native_qwen21.glb
```

`tools/generate_native_qwen21.py` = `generate_native.py --no-texture`（白模）
→ `run_guided.py --white-mesh --auto-cameras 8 --steps 20 --exponent 7.0`。

**实测（Asuka，seed 42）**

| | native（TRELLIS 自带 PBR） | 白模 + Qwen 2.1 重贴 |
|---|---:|---:|
| 耗时 | 44.4s | **352.4s** |
| 覆盖率 | 原生 100% | 47.3%（UV 岛 21.2%，岛内有效 92.7%） |
| 画质 | 红饱和、脸白块、胸橙球 | **更暗、更糊、背部斑驳** |

**结论：接上了官方路径，但本地投影拖后腿，没比 native 好。**

Qwen 原视图本身很好（`out/asuka_qwen_views_check.png`：正面有五官、红衣
橙发、T-pose 正确），坏在两处：

1. **投影是 UV texel 反向映射**（`meow3d/projection/bake.py`），覆盖率受
   UV 岛占用限制（21.2%），空洞靠 diffuse 兜底 → 暗红斑驳。
   StableGen 是 **Blender 逐面 BVH raycast 投影**，不看 UV 布局，覆盖接近满。
2. **侧/背视图与网格姿势不一致**：Qwen 把 v01/v04 画成**收臂站姿**（它按
   常识补全），而网格是 T-pose，投影必然错位。这是 README「白模路线」一节
   早已记录的老毛病。

⚠️ 也就是说：**剩余差距在投影实现，不在 TRELLIS.2、不在 Qwen 参数。**
要真正达到官方效果，得把 `bake.py` 从「UV texel 反向映射」换成 StableGen 的
「逐面 3D 表面投影（BVH × cos^k）+ Blender bake」，这是一次投影层重写。

## StableGen 投影/权重链的逐行核对（2026-11-06）

按 v0.5.0 源码逐项对过，结论是**我们的架构与它同源，但有四处实现差异**：

| 项 | StableGen v0.5.0 | 本项目 |
|---|---|---|
| 合成方式 | 一个材质内所有相机**求和÷权重和**，一次 bake（`project_image` projection.py:2248） | 逐视图烘 color/weight，numpy 加权合并（数学等价） |
| 权重 | `frustum × angle(<70°) × pow(facing,3.0) × feather × visible`（:1640-1720） | `facing^7 × alpha × 羽化 × 遮挡`，**无角度截断、无 frustum 项**（已实现为可选） |
| 可见性 | `min(Is Hit, dist < HitDistance + eps)`，射线从相机出发、`Length=dist+eps*10`；eps=bbox×1e-5（:284-305, :1550-1615） | 渲 Z pass 做深度比较（`ShaderNodeRaycast` 在本机 5.4-alpha bake 下可用，但见下） |
| BakeUV | `unwrap('pack', margin .001)` + 被投影面整体缩放填满图集、未覆盖面塌缩（:2040） | 保留/重展开后直接当 BakeUV |

**三处已修的真 bug**（都在 `tools/blender_bake_views.py`）：
1. `elif zdepth is not None` 未判遮挡模式 → 非 depth 模式也会跑深度判定，权重被清零；
2. BVH 可见性起点/方向写反（从顶点往相机退）→ 实测 93~98%「可见」= 没做遮挡；
3. ray epsilon 写死 0.02，应为 `bbox×1e-5`（≈2.6e-5），大了 500 倍。

**一个必须记住的实测事实**：本项目的 TRELLIS.2 白模是**薄壳**（`tools/_probe_shell.py`
实测平均厚度 0.005）。`ShaderNodeRaycast` 在 5.4-alpha 的 CPU bake 里是**可用的**
（球 60.7%、真网格 52.5% 有命中），但薄壳内壁 texel 的 `HitDistance` 比着色点近
0.03~0.26，被 `dist < hit+eps` 判成遮挡 → 覆盖率只剩 0.3%。这是几何属性，不是 bug。

**把参数抄过来反而更差**（已回退，别重试）：
- `unwrap=basic` 图集占用 21%→50%，但头部小岛被压扁成米色块、腿部出横条纹；
- `discard=70°` + `feather=0.15` 同样导致上述退化。
现有默认（smart unwrap / feather 0.1 / 不截断 / 深度遮挡）重测**岛内覆盖 94.2%、
UV 占用 47.5%、4096²**，画质正常。真正瓶颈在**Qwen 视图的取景**（角色只占画面
8.9%）与薄壳网格，不在权重公式。

## 取景核对：与 StableGen 的差距，以及实测结论（2026-11-06）

**StableGen 的取景**（`cameras/placement.py:955` `_create_cameras_per_aspect`）：
逐相机独立取景——`_compute_silhouette_distance(margin=0.10)` 让网格刚好占满画幅
留 10% 余量、aim 到轮廓视觉中心；画幅长宽比 = 该视角轮廓 w/h；
`_resolution_from_aspect(total_px=1MP, align=32)` → 非方形、1MP、32px 对齐。

**我们**：所有视角共用一个 `fit_ortho`（最大跨度 ×1.08）、方形 1024²。
实测网格只占 **57.7%W × 71.5%H（面积 41%）**。打开 `--fill-frame`（ortho 2.80→2.16）
后为 **74.7%W × 92.7%H（69%）**，像素密度 ×1.7。

⚠️ **但视觉上没有差别**（同 2048² 图集、同 8 视图对照，`out/asuka_frame_fair.png`）。
覆盖率 93.9% → 93.1%。**结论：取景不是画质瓶颈**，别再往这个方向调。

### 投影：逐项核对结果为等价，不需要改

| 项 | StableGen | 我们 |
|---|---|---|
| 投影 UV | 透视存齐次 `(u*w,v*w,w)`（:360），正交用 UV Project 结果 | `u=0.5+P_cam.x/ORTHO`，与正交 UV Project 等价 |
| 归一化 | 材质内 Σ(color·w)/Σw | `merge()` 的 `acc/wsum`，一致 |
| alpha | 权重 × alpha（:250） | 同 |
| 一次 bake vs 逐视图 | 一个材质一次 bake | 逐视图烘 color/weight 再合并，数学等价 |

### 本轮修掉的三个真 bug（全是 Blender 5.x / 参数没接上）

1. **`tools/blender_guidance.py` 合成器 API**：5.x 已把 `scene.node_tree` 换成
   `scene.compositing_node_group`，`CompositorNodeMapRange/Composite` 被删除。
   症状：`guidance.py` 只渲 0 个机位却**以 0 退出**，`--reuse-views` 完全掩盖它。
2. **`scene.use_nodes = False` 在 5.x 不关合成器**：深度/法线渲过后合成节点残留，
   把后面的 beauty/flat 也替换成深度图 —— 实测 beauty 均值 156 → **7.3（近全黑）**。
   必须同时关 `render.use_compositing`，已加 `_compositor()` helper。
3. **`--size` 从来没传给 bake**：白模路线的图集被 `bake_projection` 的缺省卡在
   **1024²**（StableGen 的 `texture_resolution` 是 2048）。已新增 `--atlas-size`
   （默认 2048）并接进 bake 命令；`run_guided` 里还加了「指引图缺失就报错」的检查。

### 脸部发灰不是空洞

把 `bake/mask.png`（覆盖掩码）当贴图渲出来看（`out/_coverage_probe_flat_views.png`），
整只角色包括脸**全是白色 = 100% 被投影覆盖**，只有裆部/鞋底几处细缝没盖到。
所以脸不是兜底填充，而是**投上去的多视图内容被 k=7 权重平均后发灰**——下一步该查
（a）脸部哪个相机是 winner、（b）`weight_exponent` 是否该在生成后切到 15。

## 取景核对：与 StableGen 的差距，以及实测结论（2026-11-06）

**StableGen 的取景**（`cameras/placement.py:955` `_create_cameras_per_aspect`）：
逐相机独立取景——`_compute_silhouette_distance(margin=0.10)` 让网格刚好占满画幅
留 10% 余量、aim 到轮廓视觉中心；画幅长宽比 = 该视角轮廓 w/h；
`_resolution_from_aspect(total_px=1MP, align=32)` → 非方形、1MP、32px 对齐。

**我们**：所有视角共用一个 `fit_ortho`（最大跨度 ×1.08）、方形 1024²。
实测网格只占 **57.7%W × 71.5%H（面积 41%）**。打开 `--fill-frame`（ortho 2.80→2.16）
后为 **74.7%W × 92.7%H（69%）**，像素密度 ×1.7。

⚠️ **但视觉上没有差别**（同 2048² 图集、同 8 视图对照，`out/asuka_frame_fair.png`）。
覆盖率 93.9% → 93.1%。**结论：取景不是画质瓶颈**，别再往这个方向调。

### 投影：逐项核对结果为等价，不需要改

| 项 | StableGen | 我们 |
|---|---|---|
| 投影 UV | 透视存齐次 `(u*w,v*w,w)`（:360），正交用 UV Project 结果 | `u=0.5+P_cam.x/ORTHO`，与正交 UV Project 等价 |
| 归一化 | 材质内 Σ(color·w)/Σw | `merge()` 的 `acc/wsum`，一致 |
| alpha | 权重 × alpha（:250） | 同 |
| 一次 bake vs 逐视图 | 一个材质一次 bake | 逐视图烘 color/weight 再合并，数学等价 |

### 本轮修掉的三个真 bug（全是 Blender 5.x / 参数没接上）

1. **`tools/blender_guidance.py` 合成器 API**：5.x 已把 `scene.node_tree` 换成
   `scene.compositing_node_group`，`CompositorNodeMapRange/Composite` 被删除。
   症状：`guidance.py` 只渲 0 个机位却**以 0 退出**，`--reuse-views` 完全掩盖它。
2. **`scene.use_nodes = False` 在 5.x 不关合成器**：深度/法线渲过后合成节点残留，
   把后面的 beauty/flat 也替换成深度图 —— 实测 beauty 均值 156 → **7.3（近全黑）**。
   必须同时关 `render.use_compositing`，已加 `_compositor()` helper。
3. **`--size` 从来没传给 bake**：白模路线的图集被 `bake_projection` 的缺省卡在
   **1024²**（StableGen 的 `texture_resolution` 是 2048）。已新增 `--atlas-size`
   （默认 2048）并接进 bake 命令；`run_guided` 里还加了「指引图缺失就报错」的检查。

### 脸部发灰不是空洞

把 `bake/mask.png`（覆盖掩码）当贴图渲出来看（`out/_coverage_probe_flat_views.png`），
整只角色包括脸**全是白色 = 100% 被投影覆盖**，只有裆部/鞋底几处细缝没盖到。
所以脸不是兜底填充，而是**投上去的多视图内容被 k=7 权重平均后发灰**——下一步该查
（a）脸部哪个相机是 winner、（b）`weight_exponent` 是否该在生成后切到 15。

## 重影（ghosting）的根因与顺序生成（2026-11-06）

**根因在我们自己身上**：`run_guided.py` 的 `--no-refine` 分支默认
`align="similarity"`，`make_similarity_aligned_view` 按**每张图各自的 bbox**
做等比缩放+平移。8 张生成图的 bbox 并不相同（实测宽差 8~28%），于是同一处细节
在各视图里落到不同高度，加权混合后就是「两个胸 / 双线」。

**StableGen 全程不做 2D 重对齐**：它只用相机投影 UV（`projection.py:360`），
所以天生没有这种重影。

已加 `--align-views`（默认 `frame`＝不做变换）。同一批视图只换对齐方式：

| 对齐 | 岛内覆盖 | 重影 | 问题 |
|---|---:|---|---|
| `similarity`（旧默认） | 93.1% | **有** | 变形造出双影 |
| `frame`（新默认） | 73.2% | **无** | 生成图没盖住的区域露底（diffuse 兜底发糊） |

### 顺序生成：两个近似方案都失败（别再试）

1. **一次性 context 二次生成**（渲 `ctx_render` → 品红补齐 → 8 视图重生成 → frame 投影）：
   覆盖 73.2% → **85.6%**（洞确实补上了），但**画面更花**——8 张图各自是整幅重绘，
   光照风格不一致，混合后红黑斑驳。StableGen 的顺序是**逐视图增量**，不是一锅端。
2. **frame + `--blend winner`**：73.2%，与 weighted 视觉接近，斑驳照旧
   → 证明问题不在混合方式，而在 8 张图内容本身不一致。

### 唯一正确的实现（下一步）

```
for i in 0..N-1:
    ctx_i = 渲染(当前已累积贴图 + 未覆盖区品红, 相机 i)   # rendering.py:1017 ctx_render{N}
    img_i = Qwen(ctx_i)                                  # 只改品红
    atlas = 投影(img_i, 相机 i)                           # **整幅**投影即可（图本身保留了上下文）
```
StableGen 顺序生成**不需要额外 mask**：生成的图本身保留了上下文，只有品红区被改。
`tools/blender_context_views.py`（新增）已能渲 ctx，可直接复用。

## 重影（ghosting）的根因与顺序生成（2026-11-06）

**根因在我们自己身上**：`run_guided.py` 的 `--no-refine` 分支默认
`align="similarity"`，`make_similarity_aligned_view` 按**每张图各自的 bbox**
做等比缩放+平移。8 张生成图的 bbox 并不相同（实测宽差 8~28%），于是同一处细节
在各视图里落到不同高度，加权混合后就是「两个胸 / 双线」。

**StableGen 全程不做 2D 重对齐**：它只用相机投影 UV（`projection.py:360`），
所以天生没有这种重影。

已加 `--align-views`（默认 `frame`＝不做变换）。同一批视图只换对齐方式：

| 对齐 | 岛内覆盖 | 重影 | 问题 |
|---|---:|---|---|
| `similarity`（旧默认） | 93.1% | **有** | 变形造出双影 |
| `frame`（新默认） | 73.2% | **无** | 生成图没盖住的区域露底（diffuse 兜底发糊） |

### 顺序生成：两个近似方案都失败（别再试）

1. **一次性 context 二次生成**（渲 `ctx_render` → 品红补齐 → 8 视图重生成 → frame 投影）：
   覆盖 73.2% → **85.6%**（洞确实补上了），但**画面更花**——8 张图各自是整幅重绘，
   光照风格不一致，混合后红黑斑驳。StableGen 的顺序是**逐视图增量**，不是一锅端。
2. **frame + `--blend winner`**：73.2%，与 weighted 视觉接近，斑驳照旧
   → 证明问题不在混合方式，而在 8 张图内容本身不一致。

### 唯一正确的实现（下一步）

```
for i in 0..N-1:
    ctx_i = 渲染(当前已累积贴图 + 未覆盖区品红, 相机 i)   # rendering.py:1017 ctx_render{N}
    img_i = Qwen(ctx_i)                                  # 只改品红
    atlas = 投影(img_i, 相机 i)                           # **整幅**投影即可（图本身保留了上下文）
```
StableGen 顺序生成**不需要额外 mask**：生成的图本身保留了上下文，只有品红区被改。
`tools/blender_context_views.py`（新增）已能渲 ctx，可直接复用。

## 顺序生成（sequential）：按 v0.5.0 源码逐行实现（2026-11-06）

### 源码里到底是什么

| 环节 | 位置 | 做法 |
|---|---|---|
| 主循环 | `texturing/generator.py:1848` `async_generate` | 相机 0 用参考图生成；相机 i>0 先渲 `ctx_render{i}` + visibility mask 再生成；每轮 `project_image(stop_index=i)` 把 0..i 全部相机重投进同一张 BakeUV |
| ctx 渲染 | `rendering.py:940-1010` `_export_emit_image_impl` | **把投影材质的 color 输出直接接 Emission 渲出来**，`fallback_color` 改的是那个 MixRGB 的 Color2；不是从 UV 图集采样 |
| 投影 UV | `projection.py:360` `_write_projection_uvw` | 每个 corner 存投影后的 `(u,v)`（正交直接用 UV Project 结果），材质用 attribute 读 |
| visibility mask | `rendering.py:1675-1685` | sequential 下强制 `COMPARE(1-fac, 1, eps)`，**只标未覆盖区**；`allow_modify_existing_textures` 关时已贴图区域不改 |
| 提示词 | `qwen_models.py:169-175` | *"Preserve its existing textured regions and replace every solid magenta region with texture that seamlessly continues the surrounding surface."* |

关键结论：**mask 只约束「生成时改哪里」，不约束投影**。生成图保留了 ctx 的非品红
区域，所以整幅投回去不会破坏已有内容——前提是 ctx 与投影同源、没有重采样损失。

### 用法

```bash
.venv/Scripts/python.exe tools/run_sequential.py out/native_q21_white.glb \
    --reference out/in/asuka_ref.png --out out/asuka_seq.glb \
    --cameras 8 --atlas-size 2048 --view-size 1024 --steps 20
```

`tools/run_sequential.py` = 首图（单图 + ControlNet depth）→ 逐轮
「多相机投影材质渲 ctx → Qwen 只补品红 → 整幅投影累积」→ 合并导出。
`meow3d/projection/bake.py:bake_views_only()` 只烘不合并，逐轮追加
`views/*.npy`，`merge()` 全量相加 == `project_image(stop_index=i)`。

### 四个实测踩过的坑（别再试）

1. **首图两格拼图**：`--two-images`（参考图 + 素模）会让 Qwen 输出
   「左红角色 / 右灰素模」拼图，整幅投影后满身白灰斑。实测 v00 前景 **84%**
   落在网格剪影之外。改成**单图 + ControlNet depth** 后干净。加了
   `_collage_guard`（剪影外溢 > 45% 判定）+ 换 seed 重试；只数连通域不管用
   （两小人手部相连会并成一个）。
2. **ctx 从图集采样 → 满身彩点**：2048 图集在 1024 画布上最近邻采样，输出像素
   落进不同小 UV 岛。品红也只应涂「UV 岛内 & 未覆盖」的 texel。
3. **ctx 从图集采样 → 反馈退化**：`图集 → 渲染 ctx → 生成 → 再投回图集`
   每轮二次重采样，误差累积成头发花白、腿部白块。**正解是把 ctx 换成
   `tools/blender_context_projection.py` 的多相机投影材质**（照 :360 的 corner
   attribute），渲的是各相机原始图按 `pow(max(dot(N,camDir),0), k) × alpha`
   实时求和÷权重和，未覆盖处品红。`blender_context_views.py` 保留作图集版对照。
4. **只投新增区反而更差**：v3 试过在投影端用 visibility mask 截断，覆盖率增长
   变慢、大半个模型仍只有首图信息。mask 是给生成用的，不是给投影用的。

**2D 对齐**：顺序生成只允许 `frame`（不做变换）；`similarity`/`bbox` 按各自
bbox 缩放，同一细节在不同视图落到不同高度 → 重影。

新增单测 `tests/test_seq_ctx.py`(4)、`tests/test_seq_guard.py`(4)。

### 五版全量 A/B：顺序生成跑通了，但**当前不如单轮版**（如实结论）

全部同一张 `asuka_ref.png`、seed 42、8 机位、2048² 图集、20 步：

| 版本 | 改了什么 | 结果 |
|---|---|---|
| 单轮 baseline | 8 张独立生成图一次加权投影 | **最干净**：红衣饱和、橙发完整、脸有五官 |
| SEQ v1 | 逐视图增量，但首图是两格拼图 | 满身白灰斑 |
| SEQ v2 | 首图改单图 ControlNet + 整幅重投影（ctx 取图集） | 白斑减少，头发花白、腿白块 |
| SEQ v3 | 只投「当轮品红新增区」 | 略干净，但大半个模型只有首图信息 |
| SEQ v4 | ctx 换成多相机投影材质 + 整幅投影 | **更差**：整体褐灰（Qwen 每轮重画非品红区 → 变暗） |
| SEQ v5 | 加保护性合成（mask 外硬保留 ctx） | 比 v4 红一点，仍不如单轮版 |

对比图：`out/asuka_seqv5_compare.png`、`out/asuka_seqv3_compare.png`。

**结论**：顺序链已按 `generator.py:1848` 实现并跑通（8 轮 ~9 分钟，
`inpaint 区` 10.9%→3.3% 逐轮递减），但**在本项目当前实现下不推荐作为交付**。
剩余三处结构性差异，都不是调参能补的：

1. **BakeUV 未 compact**：StableGen `_compact_uncovered_projection_uvs`
   （rendering.py:2298）把被投影覆盖的 UV 整体缩放填满图集、未覆盖整面塌缩；
   我们用 smart unwrap 在 10 万面上切出上千个小岛，bake margin 在岛间串色。
2. **不是真 inpaint**：我们的 Qwen 是整幅 img2img edit，保护性合成只是宿主侧
   近似；StableGen 的 mask 进 ComfyUI workflow，mask 外像素不进潜空间。
3. **投影材质是一次 bake 的材质内归一化**（project_image :2248），我们是逐视图
   烘 color/weight 后 numpy 合并，边界处理不等价。

> 当前**推荐交付仍是单轮 8 视图加权版**（`out/asuka_fix.glb`）。顺序链保留为
> `tools/run_sequential.py`，待上面三项对齐后再切默认。

### 五版全量 A/B：顺序生成跑通了，但**当前不如单轮版**（如实结论）

全部同一张 `asuka_ref.png`、seed 42、8 机位、2048² 图集、20 步：

| 版本 | 改了什么 | 结果 |
|---|---|---|
| 单轮 baseline | 8 张独立生成图一次加权投影 | **最干净**：红衣饱和、橙发完整、脸有五官 |
| SEQ v1 | 逐视图增量，但首图是两格拼图 | 满身白灰斑 |
| SEQ v2 | 首图改单图 ControlNet + 整幅重投影（ctx 取图集） | 白斑减少，头发花白、腿白块 |
| SEQ v3 | 只投「当轮品红新增区」 | 略干净，但大半个模型只有首图信息 |
| SEQ v4 | ctx 换成多相机投影材质 + 整幅投影 | **更差**：整体褐灰（Qwen 每轮重画非品红区 → 变暗） |
| SEQ v5 | 加保护性合成（mask 外硬保留 ctx） | 比 v4 红一点，仍不如单轮版 |

对比图：`out/asuka_seqv5_compare.png`、`out/asuka_seqv3_compare.png`。

**结论**：顺序链已按 `generator.py:1848` 实现并跑通（8 轮 ~9 分钟，
`inpaint 区` 10.9%→3.3% 逐轮递减），但**在本项目当前实现下不推荐作为交付**。
剩余三处结构性差异，都不是调参能补的：

1. **BakeUV 未 compact**：StableGen `_compact_uncovered_projection_uvs`
   （rendering.py:2298）把被投影覆盖的 UV 整体缩放填满图集、未覆盖整面塌缩；
   我们用 smart unwrap 在 10 万面上切出上千个小岛，bake margin 在岛间串色。
2. **不是真 inpaint**：我们的 Qwen 是整幅 img2img edit，保护性合成只是宿主侧
   近似；StableGen 的 mask 进 ComfyUI workflow，mask 外像素不进潜空间。
3. **投影材质是一次 bake 的材质内归一化**（project_image :2248），我们是逐视图
   烘 color/weight 后 numpy 合并，边界处理不等价。

> 当前**推荐交付仍是单轮 8 视图加权版**（`out/asuka_fix.glb`）。顺序链保留为
> `tools/run_sequential.py`，待上面三项对齐后再切默认。

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

### 白模路线（`--white-mesh`）

从 `MeshOnly` 白模出发可以做到**完全不碰 nvdiffrast / nvdiffrec**：

```bash
.venv/Scripts/python.exe tools/run_guided.py out/meow3d/asuka_WhiteMesh_00001_.glb \
    --reference out/in/asuka.png --out out/xxx.glb --white-mesh --fill-frame
```

白模 **没有 UV 也没有贴图**，开关会自动设 `protect=0` / `unwrap=smart` /
`--two-images` / **`--guide-pass clay`** / **`--align-scale height`** /
`--decimate 50000`；`refine` 检测到 flat 基准是纯白（通道差 0.003）会
**跳过颜色锚定与保护性合成**，否则会把 Qwen 生成的橙发/红战服全拉成白。
实测双图 Qwen 在白模上不拼图，颜色与细节都正确。

**为什么必须用 clay**：白模 `beauty` 是纯白，正侧视里 T-pose 的手臂退化成一条
细线，Qwen 会按常识把 v01/v03 侧视图画成**收臂站姿** —— 与 T-pose 网格对不上，
投影后手臂/胸口出现杂色脏斑。clay 素模带 cavity 明暗、轮廓清晰，重跑后侧视图
双臂水平前伸，3/4 侧与正面的斑块明显减少。纯文字约束（prompt 写死 STRICT
T-POSE）实测无效；`--blend winner`、换原生 xatlas UV、`--no-align`、45° 斜角
机位也都无效或更差，详见差距文档第十三节。

**为什么默认 `height` 对齐尺度**：`make_similarity_aligned_view` 原来用面积比
`sqrt(ref/src)` 当尺度初值，clay 侧视图双臂水平伸出后前景面积变大 → 尺度算小、
对齐 IoU 掉到 0.40。改用包围盒高度比（躯干高度稳定）后 v01 IoU
0.573→0.631、岛内有效覆盖 82.1%→85.8%，侧视胸口杂色团几乎消失、背面纹路
清晰。`--core-iou`（只用腐蚀后的躯干核心算 IoU）实测无效，默认关、不建议用。
`--align-scale area` 可回退旧行为。详见第十四节。

严格 A/B（同一批 Qwen 出图，一组下采样成等效 2.8 构图）：
高频能量 0.0404 → **0.0434（+7.4%）**，是带贴图路线（+3.0%）的 2.5 倍 ——
白模没有可信底图可锁，画幅收益确实更直接。`--white-mesh` 会自动降面到
5 万（`--decimate 0` 关闭）：UV 岛占用 14.7% → **28.2%**、GLB 51.6MB →
**4.9MB**、岛内有效覆盖 79.0% → **89.2%**（默认 `height` 对齐后 **92.7%**）。

### ⚠️ 覆盖率会骗人：看「岛内有效覆盖」

`--margin`（bake 边缘扩张）能把全图覆盖率从 26.6% 一路推到 89.4%，
但四视角预览**几乎逐像素不变** —— 扩张色落在 UV 岛外的空白 texel，
渲染时没有三角形采样。新增 `meow3d/projection/uvstats.py` 解析 GLB 的
`TEXCOORD_0` 光栅化 UV 岛，`bake` 现在会同时打印：

```
  视图 4 个，覆盖 55.1%
  真实覆盖：UV 岛占用 28.1%，岛内有效 89.3%，岛外虚高 41.7%
```

判断画质请只看 **岛内有效覆盖**；`margin >= 16` 且虚高过大时会明确警告。
实测否定结论：`cube` UV 占用翻倍但重叠串色（头发/脸出现红斑），
`2048²` 反而让 UV 占用从 14.7% 降到 12.4%；`cube+2048` 与降面 <3 万
会让 Blender 崩溃（exit 11），CLI 已提前提醒。

侧视杂色条纹的真正根因是 **Qwen 把 T-pose 侧视画成收臂站姿**（见第十三节），
已由 `--guide-pass clay` + `--align-scale height` 大幅缓解；剩下的少量残留
要靠 depth/normal 特征配准或 Local Edit 局部修，而不是继续调 UV/margin。
详见差距文档第十二～十四节。

全量单测 **65 passed**（`tests/test_align_scale.py` 新增 3 条）。

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
