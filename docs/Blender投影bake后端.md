# v2 投影后端：Blender 3D 表面投影 + bake

> 实现时间 2026-11-04。按 StableGen Wiki 公开的**架构思路**独立实现，
> 未复制其 GPL-3.0 源码；整条链不调用 `nvdiffrast` / `nvdiffrec_render`。
> 代码：`meow3d/projection/bake.py`（宿主）、`meow3d/projection/glbpatch.py`
> （二进制换贴图）、`tools/blender_bake_views.py`、`tools/blender_apply_atlas.py`、
> `tools/bake_projection.py`。
> 诊断工具：`tools/compare_heads.py`（头部特写对比，`--flat` 看贴图本色）、
> `tools/blender_preview.py`（四视角 + 头部特写，`flat` 参数切平光）。

## 一、为什么必须重写（v1 必碎）

v1（`meow3d/projection/geometry.py` + `pipeline.py`）在 **UV 图集 texel 上**做投影：
每个 texel 反查 3D 点 → 按 `cam.screen()` 采样视图。TRELLIS.2 原生链用 xatlas
展开出 **106 个碎片 UV 岛**，19 万个微小面只写了中心像素；`points` 在岛与岛之间
跳变，相邻 texel 会采到视图里完全无关的位置 → **满屏碎玻璃**。

StableGen 的做法是反过来的：投影是**渲染时的 3D 表面材质**（Cycles + raycast OSL），
UV 贴图只是最后 `bake` 出来的产物。官方 Wiki *Export* 原话：

> "A StableGen texture is a **projection material** that combines the camera
> images **at render time**. To use the result elsewhere, **bake it into
> ordinary UV textures**, then export."

投影在连续表面上完成，三角形内部靠重心插值，压根不存在「texel 跳变」。
本后端就是把这件事用 Blender 原生能力重做一遍。

## 二、新架构（4 步）

```
① 宿主侧  视图严格对齐
   读 GLB 顶点 → 按相机矩阵算模型投影 bbox → 把视图主体 bbox 映射过去
   输出 RGBA 方形 PNG（alpha=0 的区域排除权重，背景色不会混进贴图）

② Blender  3D 表面投影 + bake（CYCLES）
   归一化（与 preview 同一套）→ UV 策略 → 每个相机生成 ProjUV
   颜色 bake：Emission(视图颜色, vector=ProjUV) → BakeUV
   权重 bake：Emission(alpha · max(N·V,0)^k) → BakeUV
   → views/<name>_color.npy / _weight.npy

③ 宿主侧  numpy 合并
   weighted 或 winner；未覆盖 texel 用原生贴图 base 兜底 / 邻域扩散
   → atlas.png（线性 → sRGB、行翻转）

④ 出口两条路（UV 是否变了决定用哪条）
   unwrap=keep（UV 没变）→ **二进制替换原 GLB 的 baseColorTexture**，网格一字节不改
   重新 unwrap           → Blender 贴 atlas 到 BakeUV 导出新网格
```

**关键：bake 是「逐三角面正向光栅化」，不是「逐 texel 反查」。** 这一条
方向的差别就是碎与不碎的分界。

### 为什么出口必须分两条（实测数据）

| 出口 | 顶点数 | 文件 | metallicRoughness |
|---|---|---|---|
| 原 GLB | 362,229 | 21.0MB | 有（2048²） |
| Blender 往返导出 | **1,351,812** | **50.9MB** | 保留 |
| trimesh 重导出 | 362,229 | 14.8MB | **丢了** |
| **二进制替换（现在默认）** | 362,229 | 21.0MB | 保留 |

- Blender 导入按 **loop 拆顶点**（UV 缝、硬边各拆一份），顶点涨 3.7 倍。
  keep 模式下 UV 根本没变，这份绕路纯属白花磁盘。
- trimesh 重导出顶点数虽然对，但**静默丢掉 `metallicRoughnessTexture`** ——
  金属/粗糙度外观直接变了，比体积问题严重得多。
- `meow3d/projection/glbpatch.py` 只改 BIN chunk 里那张图的字节：
  定位 `baseColorTexture → textures[i].source → images[j].bufferView`，
  替换时**连同老图后面的对齐 padding 一起吃掉**（否则长度增量不是 4 的倍数，
  后面所有 bufferView 的 `byteOffset` 全部错位），再平移后续偏移。
  21MB 文件 **0.03 秒**改完。

图集边长默认 `--size 0` = **自动等于原生贴图边长**（本例 2048）。给 1024 时
投影区只剩原生 1/4 的纹素，是「糊」的主要来源之一。

### ProjUV 怎么算（不用 operator，全 numpy 向量化）

```python
inv = np.array(cam.matrix_world.inverted())
P = V @ inv[:3,:3].T + inv[:3,3]      # 相机空间
u, v = 0.5 + P[:,0]/ORTHO, 0.5 + P[:,1]/ORTHO
proj.data.foreach_set("uv", np.stack([u[lvi], v[lvi]], 1).ravel())
```

`lvi` 是 loop→vertex 索引。三角形内部的投影坐标由 bake 光栅化插值，连续。

## 三、实测（RTX 4090 / Blender 4.5.9）

| 版本 | UV | 图集 | 视图 | 覆盖 | 结果 |
|---|---|---|---|---|---|
| v1 texel blend（旧） | xatlas 碎片岛 | 1024 | Qwen 4 视角 | — | **碎玻璃** |
| v2 单相机 · Cube UV + 扩散 | Cube Project | 1024 | 原图 | 14.9% | 不碎，但正面斑驳、侧背被染成正面图案 |
| v2 单相机 · 原 UV + base | 保留不重叠 UV | 1024 | 原图 | 52.0% | 干净锐利，侧背保留原生贴图 |
| v2 四相机 · 加权 | 原 UV | 1024 | Qwen 4 视角 | 98.1% | 不碎，正面略糊、侧面有斑驳 |
| v2 从白模起步 | Smart UV（降面 10 万） | 1024 | Qwen 4 视角 | 34.7% | 不碎、人物完整，不用原生贴图 |
| **v2 单相机 · 2048** | 原 UV | **2048** | 原图 | 45.0% | **最干净**，正面像素级还原，侧背原生 |
| v2 四相机 · 2048 + 遮挡剔除 | 原 UV | 2048 | Qwen 4 视角 | 91.0% | 覆盖高，仍有侧视污染（见 §六） |

2048 跑完只要 **17 秒**（4 相机含 4 次深度渲染 + 8 次 bake + 导入）。
对比图 `out/final_three_way.png`（原生 / 单相机 / 四相机+剔除）、
头部特写 `out/heads_lit3.png`、`out/heads_occ_flat.png`。

### 投影是在"还原设计"，原生链是在"发明细节"

同一张参考图，两条链的行为完全不同：

- **原生链**（TRELLIS 生成贴图）：细节锐利，但脸部是**编的** —— 参考图是露脸
  骑士，它把脸生成成了一块金属；嘴、眉、眼全是重绘的。
- **投影链**：脸部皮肤色、眼睛、眉毛、金边位置都与参考图一致（平光下量化
  中位色 156,150,135 vs 158,152,146，几乎没有色差），代价是**高频糊**。

结论：投影负责"像不像"，原生负责"细不细"。投影糊的根子是参考图分辨率
（全身图里脸只有 ~160px，投到 2048 图集等于放大 1.4 倍以上），不是图集不够
—— 实测 UV 利用率 53%，头部占图集 10%（≈647² 等效），纹素预算充足。

`--detail`（频率分离）是补这个缺口的折中：颜色仍用投影，把原生贴图的**高频**
（金属浮雕、布料纹理）按权重加回去 —— 两者共用同一套 UV，像素天然对齐，
不会出双边缘。实测头部高频能量 10.38 → 11.10（原生 9.46），肉眼是"浮雕清楚
一点"，0.3~0.6 安全，`>=1` 开始出脏边。

## 四、踩过的坑（都是「不报错但结果错」）

1. **glTF Y-up vs Blender Z-up**：trimesh 读到的是 glTF 原始坐标（Y 是高度），
   Blender 导入绕 X 轴 +90° 转成 Z-up。不转换时算出的模型 bbox 只有
   `v=[0.404,0.596]`（厚度），视图会贴歪。转换 `(x,y,z)→(x,-z,y)` 后 host 的
   `(0.192,0.143,0.808,0.857)` 与 Blender 打印的 `u=[0.192,0.808] v=[0.143,0.857]`
   **完全一致** —— 这是视图与相机严格对齐的判据。
2. **`export_uvs` 已失效**：Blender 4.5 glTF 导出参数叫 `export_texcoords`。
   三个 Blender 脚本都要改。
3. **EEVEE Next 不支持 `bpy.ops.object.bake`**：报
   `Current render engine does not support baking`，必须用 **CYCLES**（1 sample 足够
   烘 Emission，实测几秒）。
4. **Cube Project 会 UV 重叠**：同一方向的多个面（披风前层 / 躯干 / 背后）投到
   同一区域，bake 互相覆盖 → 表面斑驳。**bake 需要的是「不重叠」，不是「大块」**；
   碎片 UV 岛对正向光栅化完全无害。所以默认 `--unwrap keep` 保留 TRELLIS.2 的
   xatlas UV。Wiki 推荐 Cube Project 是针对「渲染时材质」，不是 bake 容器。
5. **单相机不能靠扩散填侧背**：会把正面图案糊满全模型。正确做法是
   `--fill base`：直接读输入 GLB 的 `baseColorTexture` 当未覆盖 texel 的底图。
   注意只有 `--unwrap keep` 时底图与 BakeUV 同坐标系。
6. **色彩空间与行序**：视图 PNG 是 sRGB，Blender 加载转线性 → Emission bake 输出
   线性 → 合并后必须 `_linear_to_srgb` 再存 PNG。`image.pixels` 从**左下角**开始，
   PIL 首行在上，输出要 `np.flipud`；底图也要 flip 才能和 bake 结果对齐。
7. **`_fill_holes` 覆盖 winner 结果**：无底图时曾用扩散图整体替换 atlas，把
   winner 的单视角颜色悄悄换回加权平均。`test_merge_winner_*` 钉住了这个 bug，
   现在只填 `~filled` 区域。
8. **不要用颜色是否为 0 判断空洞**：纯黑贴图会被误判成未填充。必须单独维护
   `filled = wsum > 1e-4` mask。
9. **视图要带 alpha**：Qwen 出图是浅灰底，直接投影会把背景混进权重。用
   `tools/matte.py`（BiRefNet）批量抠成 RGBA，Blender 侧 `tex.Alpha` 乘进权重。
10. **必须做遮挡剔除**：不做的话，被挡住的背面会把颜色投到正面 —— 侧视图的
    蓝色羽饰糊在银白头盔上，看起来像"脏斑"。做法（比 StableGen 的 OSL raycast
    便宜得多）：同一台正交相机渲一张归一化 Z pass 深度图（0=近 1=远，
    16 位 PNG，Compositor 的 Map Range 把 [2.0, 5.0] 映到 [0,1]），另一路把
    「该 texel 的视线深度」也 bake 出来，在 shader 里比
    `own - front < eps` 判可见。深度图必须 `interpolation="Closest"` +
    `colorspace="Non-Color"`，否则边缘插值会造出假遮挡。
    **代价只有每视图一次 0.7s 渲染**，覆盖从 91.6% → 91.0%（几乎不损失）。
11. **别用 trimesh 重导出 GLB**：会静默丢掉 `metallicRoughnessTexture`。
12. **别为了"干净"去 Blender 往返导出**：keep 模式 UV 没变，顶点却从 36 万
    拆成 135 万、文件 21MB → 51MB。
13. **图集给 1024 就糊**：默认跟原生贴图同边长（`--size 0` 自动）。
14. **"脸发白"可能是预览灯光过曝，不是贴图错**：白铠甲 + `metallicFactor=1`
    在 5 盏灯下直接死白，五官被冲掉。诊断必须用 `--flat`（Emission 直出贴图本色，
    不打灯），`tools/compare_heads.py --flat` 一键出对比。
15. **Blender `-b` 的 cwd 不是调用者的 cwd**：脚本里传相对输出目录会写到
    `C:\`。预览脚本现在统一转绝对路径。

## 五、当前结论

- **架构问题已解决**：投影作用域从 UV texel 搬到 3D 表面，「碎玻璃」消失。
- **默认交付配置**：`--unwrap keep --fill base --size 0`（自动 2048）+ 单张高清
  透明原图 + `--detail 0.4`（可选）。正面是原图像素级直贴（GD-Head / Headshot 3
  那套「可见区直取」），侧背保留原生贴图，不动几何、不训练、适配二次元。
- **多相机已能"不碎"，但还不够"可信"**：开了遮挡剔除后覆盖 91%、不再有明显
  大块污染，但仍有残余脏斑 —— 根子在 Qwen 生成的视图**几何不严格对齐**
  （`view_02` 号称 90°，实际还是偏正面），投影到窄侧视 bbox 里会被大幅拉伸，
  蓝色羽饰就糊到银盔上。剔除解决的是"被挡住的面"，解决不了"视角本身不对"。
  → **已补：见 §七 指引图。**
- **商用链已跑通**：从白模 + Smart UV + Qwen 4 视角 + 扩散填洞，完全不依赖
  原生链的贴图，效果可用。但本次白模仍取自完整工作流的 `WhiteMesh` 导出；
  要严格声明不碰 `nvdiffrast`，应改用 `--workflow MeshOnly.json` 重新出白模
  （`--plan` 已验证 18 节点全可映射、不含 `MeshTexturing`）。

## 六、剩余缺口（按性价比）

1. ~~Guidance Map~~ → 已实现，见 §七。
2. **UV 利用率**：53%，头部占 10%。Smart UV 路径只有 34.7%，重 unwrap 时
   应优先 `island_margin` 小 + 打包优化，而不是换投影算法。
3. **MeshOnly 白模重跑**：把商用链的源头也换成不含 NVIDIA 库的工作流。
4. **相似变换只有「尺度+平移」**：没有旋转/透视项。Qwen 若画歪了一点头颈，
   投影也会跟着歪一点。真要更准，需要特征点（眼/肩/脚）而不是轮廓 IoU。

## 七、指引图：让 Qwen 的新视角几何正确

**问题的真身**：多相机脏斑不是剔除不够，是**源视图本身角度不对**。
`view_02` 宣称 90° 右侧，实测还偏正面（宽高比 0.6 而不是 0.25），投影到窄侧视
bbox 就被横向拉伸 —— 剔除再多也救不回来。

### 7.1 做法

```
网格 ──guidance.py（同一台正交相机）──> v00_depth/normal/clay/beauty.png
                                  │
              参考图 + 彩图 ──qwen_guided_views.py──> 新视角（几何被钉住）
                                  │
                     matte.py ──> 投影 bake（--align similarity）
```

`tools/blender_guidance.py` 用**与 bake 完全相同的相机**（ORTHO 2.8、距离 3.5、
模型归一化到 [-1,1]）渲四通道，所以图里模型占的像素范围就是
`bake.projection_bbox()` 算出的投影框 —— 不需要二次对齐：

| 通道 | 用途 |
|---|---|
| `depth.png` / `depth_c.png` | ControlNet depth 条件；`_c` 是按前景分位拉满动态范围的版本（固定 [2,5] 区间里模型只占一小段灰阶） |
| `normal.png` | 法线 RGB，排查几何 |
| `clay.png` | workbench 素模 + cavity，可做人眼检查或 lineart |
| `beauty.png` | 带原贴图的彩图，`--mode beauty` 的底图 |

### 7.2 三种模式

| 模式 | 输入 | 实测 |
|---|---|---|
| **`beauty`（推荐）** | 彩图渲染 → 指令改图 | 10~12s/张，几何 100% 正确（底图就是从真 3D 渲的） |
| `control` | depth 图 + 参考图 → ControlNet Union | **24G 卡上不可用**，见 7.5 |
| `both` | 深度 + 彩图 + 参考图 | 同上，显存先炸 |

### 7.3 构图漂移必须修：`--align similarity`

Qwen 会改构图。实测侧视：主体高占比 **0.855 vs 指引 0.703**、中心偏左 **0.10 画幅**，
但**宽高比几乎一致（0.243 vs 0.246）**。所以正确做法是**等比缩放 + 平移**，
不是老的 `bbox` 模式那样 x/y 各自拉满 —— 后者会把本来对的形状捏坏，侧视尤其
致命（bbox 只有身高 1/8 宽）。

尺度初值取**面积比** `sqrt(A_ref / A_gen)`（对轮廓上多出的披风边缘/影子不敏感，
高度比对它极敏感），再在 1/4 分辨率上做两级 IoU 搜索定平移。实测 IoU
0.66~0.83、覆盖 91.4%，比无引导版的侧面干净一大截。

⚠️ **单位坑**：源图是 768²、画布是 2048²，直接拿两边像素数算比值会差
`(size/src)²` 倍 —— 侧视被缩成 0.18 倍，投影直接空掉（`BAKE view v01: 覆盖 0.0%`）。
所有面积/长度必须先在 `size` 画布上量再相除。

### 7.4 「两格拼图」：Qwen 的隐藏行为

给**两张**全身站姿图（参考图 + 彩图）时，Qwen 会时不时输出**左右两格**
（左原图 / 右结果）。汉服角色两次都中招，骑士两次都没事 —— 取决于内容而非参数。

- 默认**只给一张图**（彩图渲染）：`--single` 已是默认行为，`--two-images` 才给两张。
  单图版汉服背面从"竖条纹糊块"变成干净的背影。
- 兜底：`bake._split_side_by_side()` 会检测「两个各占 25% 面积、x 区间不相交」
  的连通块并按宽高比挑目标那个。**飘带一重叠就失效**（汉服背面正是如此：
  左图的纱带伸进右图身体，x 区间相交），所以它只是兜底，不是正解。

### 7.5 ControlNet Union：24G 卡上别碰

`models/qwen/controlnet/qwen21_union.safetensors` 是 7.55GB 的 bf16 分支。
挂上去后显存 24.0/24.5GB，单张 768²/8 步跑 **14 分钟**还在挪显存（GPU 100%
但进程查不到 CUDA 上下文）。40G+ 的卡才值得试。

顺带修了一个更隐蔽的：`tools/qwen_views.py` / `qwen_guided_views.py` 曾把
`MEOW_QWEN_OFFLOAD` 默认设成 `0`（"全程常驻，每张省 1~2s"），在 24G 上这就是
**显存满、一直转、不报错**的直接原因 —— transformer-fp8 + text-encoder-fp8
常驻 15.5GB，峰值 16.4GB。已改回主项目默认 `1`，单张从"卡 4 分钟以上"回到 12s。

### 7.6 实测

| 版本 | 视图 | 覆盖 | 结果 |
|---|---|---|---|
| 原生链 2K | — | — | 锐利，但脸/纹样是"发明"的 |
| 单相机投影 2048 | 原图 | 45% | 正面最干净，侧背是原生贴图 |
| 四相机·无引导 | Qwen text2img | 91% | 侧面蓝灰污染、头盔脏斑 |
| **四相机·引导+相似对齐** | Qwen 指令改图 | **91.4%** | **侧背干净，纹章完整，无污染条纹** |

对比图 `out/knight_final_compare.png`、`out/hanfu_final_compare.png`，
头部特写 `out/heads_guided.png`。

## 八、许可

本后端只用 Blender（GPL，作为外部工具调用、不链接其代码）、trimesh、cv2、
numpy、PIL，以及 TRELLIS.2 的权重与节点包（MIT）。**不调用 nvdiffrast /
nvdiffrec_render（NVIDIA Source Code License，禁商用）**。
Qwen Image 2.1 底模仍是 RESEARCH LICENSE，商用前需一并解决，详见
`README.md`「商用许可风险」与 `docs/StableGen评估.md`。
