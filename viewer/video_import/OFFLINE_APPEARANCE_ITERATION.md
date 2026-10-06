# 离线多视频外观迭代：2026-10-01

## 2026-10-04：补查肩背帧与衣缘复核（发现高置信度细肩带误分）

`output/offline_iterations/20261004_shoulder_evidence_v1/` 新增九段225帧、438个视图的
原像素检查。入口 `audit_shoulder_evidence.py` 每10帧取第2帧，排除原TRAIN、留出及
137/213；本轮为来源发现，不是独立评估，也非2242帧穷尽搜索。固定原几何/相机，按
原0.85置信度、类别内部3像素、正面余弦0.3、25mm深度条件检查35704个未解决纹素。
10003个至少找到一次自动皮肤/衣物候选，25701个仍无；原19040个零观测中635个有新增
候选。仅新增样本内823个通过8次、2片段、80%类别一致条件；不能据此宣称已准确恢复。

36组对照及真人/镜中九片段概览见 `review.html`、`review/`。发现交叉细肩带被预测为
皮肤，左肩及腰部又有上衣类别扩张。原生像素定点复核：48.43/f102/mirror白肩带
(1792,103)、(1798,112)被预测皮肤，置信度90.59%、89.02%；50.03/f182/real
(1521,386)、(1516,400)同类错误89.80%、94.12%，这些点部分进入自动候选。
以上为视频原坐标、0-based帧，助手视觉诊断而非用户人工真值；记录在
`source_pixel_spot_checks.json`。因此类别内部安全距离无法防止漏检细肩带，降低置信度
不合适。自动衣缘5像素邻域66.42%置信度低于0.85，也不是人工错误率。

纹素数量会严重重复原像素：49.33/f102/mirror的2180个候选仅对应16个不同取整投影像素。
没有烘焙新纹理或修改原工程；下一步先恢复细肩带和衣缘分层、复核头发边界未知带，再
重新筛选来源并按固定留出评估。几何/相机投影误差仍未独立排除，全身及光照分离暂后置。
新增3项测试和既有6项语义测试通过，输入及缓存SHA复核见 `verification.json`。

## 2026-10-04：头发/皮肤/衣物分层与衣缘来源修正（未通过发布）

`output/offline_iterations/20261004_uv_semantics_v1/` 完成九段705个TRAIN视图及89个
评估视图的本地SegFormer ONNX自动解析。入口 `semantic_body_layers.py` 使用原视频人物
裁剪、512输入，保留原像素坐标和模型/视频/缓存SHA。模型revision为
`cb6ac44e641faa309f29561c1afe08d87ef52631`，权重SHA见实验记录。模型不是人工真值；
部分腰部皮肤、短裤及球拍被误判上衣，因此针对本组白色上衣增加颜色否决，将疑似错判
退回未知。未重新训练模型，没有调用外部图片推理服务。

`refine_texture_semantics.py` 冻结几何、相机、曝光和2K分辨率，在肩背代理区根据TRAIN
皮肤/衣物共识重选真实像素。共修正33015个纹素，其中32120个原来源被标为头发；
头发污染候选修正率47.36%，未达到预设50%实验门槛。另有35704个疑似头发遮挡纹素
保留原显示值并明确标记；其中19040个在当前TRAIN和自动条件下没有合格非头发观测，
不能据此宣称全部原视频帧都没有信息。没有生成隐藏皮肤，也没有独立马尾几何。

显示覆盖率保持98.52709%；仅扣除本轮已标记肩背头发污染后的支持率96.77406%→97.60426%，
并非全材质准确率。`audit_semantic_texture.py` 同时保留原始ROI和语义一致可见域，
候选丢失alpha按255计入固定分母。137/213帧的镜中语义一致可见域MAE为7.91219→7.44999，
镜中衣缘5像素邻域11.37770→11.00989；真人衣缘略增约0.035。镜中原始ROI误差升高，
因为原照片仍有遮挡马尾，而候选部分恢复其他帧拍到的皮肤。该分域依赖同一个自动解析器，
不能作为独立分割真值或人工衣缘位移验收；137/213帧此前已用于光度实验。

25项相关测试通过，固定Blender对照显示残留黑块与新皮肤边界仍明显，故未更新Viewer、
原九段工程，也未扩展全身/进行光照分离。下一步先补查未入TRAIN的无遮挡肩背帧与低置信度
分割、衣缘；确认长期遮挡后补充观测，并考虑独立头发表示。原图分层、UV局部类别掩码、
未解决遮挡图和全部决策见实验目录 `REPORT.txt`、`decision.json`、`layer_review/`。

## 2026-10-04：肩背颜色与接缝消融（局部候选保留）

`output/offline_iterations/20261004_uv_photometry_v1/` 完成九段2K固定对照。
`refine_texture_photometry.py` 只在TRAIN自动筛选的肩背区域估计逐来源有界BGR偏移，
并比较基线、仅颜色校正、颜色校正加窄带屏蔽Poisson。它不是完整MVS顶点接缝校正，
也不是反照率/光照分离；没有改变几何、来源、透明度或手部。

颜色校正两组未通过留出回归。补充 `ablate_texture_seam_only.py` 去掉颜色校正后，
保留相同接缝参数，来源边界色差代理从4.42460降至1.15515；原留出肩背MAE
真人12.05607→12.02953、镜中12.70226→12.69732，覆盖率保持98.52709%。
`audit_texture_fresh_frames.py` 固定候选后，在九段原纹理TRAIN/留出缓存均未包含的
第137、213帧检查：真人8.86257→8.83348、镜中9.02721→9.02427。
这些仍是同次拍摄、估计几何下的颜色指标，接缝代理同时是优化目标，不能当独立精度真值。

19项相关测试通过；49个源数据/缓存/基线文件SHA一致。48.43、48.53固定姿态相机灯光
共16张Blender对照显示局部接缝略有缓和，但背部头发遮挡、衣缘和肩部色块仍明显。
只保留 `poisson_only` 为局部候选，未替换Viewer或原九段Blender材质，未扩到全身。
原图马尾遮住肩背，单一人体轮廓遮罩不能识别这一遮挡；下一步优先验证头发/皮肤/衣物
分层及肩带、衣缘对应。旧缓存生成绑定限制仍保留。完整协议、数值、原像素裁剪和决策
见实验目录 `REPORT.txt`、`decision.json` 与 `fresh_frames/`。

## 2026-10-04：肩背/手部可见性候选（未通过）

独立实验位于 `output/offline_iterations/20261004_uv_visibility_v1/`，决策为
`completed_rejected_keep_delivered_material`。入口 `refine_texture_visibility.py`
按 prepare / fuse / evaluate 执行。保持2K、几何、相机、TRAIN颜色、旧曝光增益和留出帧不变，
候选采用有向法线（恢复一次镜中绕序）和手部10mm/其他25mm深度容差；尚未拆分这两个因素。

原TRAIN手部代理观测中，真人36.58%、镜中26.07%为背向面。源视频2560×1440，
存储遮罩512×288，不能把后者上采样当成高精度边界。九段原生archive哈希与既有
真人/镜中MHR回放报告一致；此次未改变任何人体参数，也未重新拟合几何。

同一正面固定留出子集，真人手部共同颜色MAE均值16.6131→16.6131，镜中21.2938→20.8198。
缺失按255计入固定分母后，分别16.6467→18.6565、21.7867→31.0385，均退步。
UV覆盖98.5271%→95.7316%；接缝内部采样代理的手部共同均值5.5171→6.7707。
原始全留出域和前向固定域均保留，不利用删掉困难样本宣称整体改善。

相同帧/相机/灯光的Blender候选仅保留复查，原9段工程和114个源文件哈希不变。
5项可见性/接缝测试与9项既有纹理测试通过。曝光只做TRAIN颜色变化诊断，未改曝光；
材质和光照分离尚未执行，因为前置对应关系候选未过关。下一步先拆开法线/深度容差对照，
补足原图ROI遮罩与可见关节对应，再做有界几何/投影修正。原缓存生成绑定限制仍保留。

完整指标、门槛、来源裁剪、代码版本、自动ROI队列和复现说明见该目录的
`REPORT.txt`、`decision.json`、`heldout_comparison.json` 和 `code_snapshot/`。

本轮使用三个 agent：纹理像素回退、分组训练输入清单、独立只读审查。只运行本地 CPU；复用已有九段 SAM3D/SAM2 重建和观测缓存。当前 Viewer、云 GPU、既有候选与验收记录均不发布或改写。

## 两组输入

缺省视频和新增九段视频是两个独立的人物/服装/球拍组。固定环境不能作为共享身体材质的依据。scene_id 可共用；person_id、outfit_id、racket_id、group_id 分开。标签根据用户分组声明，未做自动身份识别。共享相机/镜面标定仍需独立验证。

`appearance_training_manifest.py` 使用显式身份标签、文件 SHA、实际视频帧数/分辨率、MHR 拓扑/UV、原生参数与相机字段建立 TRAIN/HELDOUT 输入。默认每 5 帧抽样，每 25 帧留出；回退以清单 TRAIN 集合为准；自定义划分不得重叠，同一视频的别名也不得泄漏。此清单只证明离线输入完整性，不证明原生 GPU 回放、几何、光照或视觉验收。

九段组具备完整输入；缺省组缺少完整原生重建及相机元数据，标为 needs_input。球拍资产单独登记，本轮未绑定版本，不能称球拍优化完成。训练清单只包含真人视图；现有融合的独立镜面来源仍由镜面缓存校验控制，不继承训练清单的验收。

## 纹理像素回退

旧方法为每张面选择一个来源，面内纹素若被遮挡就透明。新开关 `--pixel-fallback` 为透明纹素尝试最多 1–8 个正分 TRAIN 次选来源（默认 3），重新检查真实表面点的遮罩和深度，仅采样原视频颜色。已有有效纹素保持原来源。

`texture_sources.npz` 记录逐像素 clip/frame/view；新增 fallback_rank：0 为主源，1…K 为回退，-1 为未观测。来源帧使用 int32。未知纹素透明；gutter RGB 只是过滤边缘，不计入覆盖。排序和表面采样分块，解码与重建一次保留一段，不缓存全部视频帧。

回退必须提供已校验的 `--group-manifest`，绑定实际融合的视频、metadata、native 与 UV，并检查缓存划分。旧默认融合仍允许不传清单，不能称所有入口都已强制隔离。

新生成的观测缓存同时绑定 NPZ 内容和生成时 mesh metadata 的 SHA；文件变化即拒绝。已校验分组融合复用旧缓存时默认拒绝，只有显式 `--allow-legacy-cache` 才可兼容，报告标注 legacy_unverified_generation_binding。现有九段缓存没有历史 metadata/NPZ 绑定，因此本轮小范围对照使用该明确兼容模式；运行时对输入及缓存哈希的绑定不等于追溯验证了旧缓存生成条件。

回退后的面中心 heldout RGB MAE 仍只评估主选源，不代表 atlas 像素的留出误差。有限次选搜索后的透明也不证明所有视角均不可见。覆盖增加不等于纹理接缝、人物几何或新视角质量通过。

## 借鉴 GitHub 项目

下一阶段优先借鉴 [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec) 的固定几何、材质和光照分离优化；需要将静态 mesh 输入改成每帧 MHR 顶点与共享 UV。借鉴 [SPARK / MultiFLARE](https://github.com/KelianB/SPARK) 的跨视频共同材质与每段光照组织，不迁移其 FLAME 人脸模型。

本轮没有复制第三方源码、增加渲染器或训练模型。清单为后续适配准备接口，不能称已完成可微材质优化。GPU 开启后应先小组试验：固定原生几何，训练 RGB 材质/每段光照，固定留出帧；报告像素残差、覆盖、接缝代理、输入和产物 SHA，并进行多角度视觉审查，再决定扩到九段。

## 本轮复现

先生成输入清单，输出必须是新路径：

```bash
python3 -B viewer/video_import/appearance_training_manifest.py \
  --group-spec output/offline_iterations/20261001_texture_v2/nine_group_spec.json \
  --layout output/multivideo_texture/mhr_uv.npz \
  --output <new_training_manifest.json>
```

独立新输出的融合可通过 multivideo_texture.py 的 CLI 开启回退。真实小范围对照只使用前两段、512 atlas、固定已有 consistency 观测；512 用 Python fuse API（CLI 支持 1024/2048/4096）。最终指标和来源保护证据见 `output/offline_iterations/20261001_texture_v2/comparison_report.json`。这是一轮候选实验，未发布 Viewer。

## 真实小范围对照结果

| 项目 | 主选源 | 像素回退 |
|---|---:|---:|
| 两段、512 atlas 的有效纹素 | 216,991 | 222,044 |
| 模板 UV 内覆盖 | 89.8435% | 91.9357% |

新增 5,053 个有效纹素（+2.0922 百分点），旧有效纹素损失 0；旧 RGBA、clip/frame/view 来源及 UV face map 完全相同。新增 rank 在 1–3，全部有效来源位于清单 TRAIN 且在冻结观测中有正分表面证据。Viewer 五项保护文件 SHA 前后相同。22 项输入/绑定测试和 9 项纹理测试通过，独立 agent 最终审查无阻断问题。

本轮旧缓存采用明确兼容模式，不能证明生成时 metadata 哈希绑定；也未评估回退像素的 heldout 投影误差。只完成两段候选实验，未将改善外推为九段验收通过。

根目录 Graphify 重建已运行；另重建仅包含 Git 管理源码（含本轮新增文件）的图，排除忽略的输出与依赖。`graphify-out/` 为本地派生产物。

复现 Python API 核心参数如下；应选择新的输出路径，不覆盖本轮报告：

```python
from pathlib import Path
from multivideo_texture import fuse  # 将 viewer/video_import 加入 sys.path
root = Path('output/offline_iterations/20261001_texture_v2')
fuse(root / 'smoke_batch', Path('output/multivideo_texture/mhr_uv.npz'),
     Path('<new_candidate_directory>'), size=512, step=5,
     mirror_source='independent', selection='consistency',
     observation_root=Path('output/multivideo_texture/fusion_independent'),
     pixel_fallback=True, fallback_max_candidates=3,
     group_manifest=root / 'nine_training_manifest.json',
     allow_legacy_cache=True)
```

## 九段全量验证：2026-10-02

同一九段、同一固定观测、2048 atlas 对照已完成。主选源 96.6384% → 回退 98.5271%（+1.8887 百分点）；新增 73,072，旧有效纹素损失 0，原 RGBA 与来源完全保持。旧基线逐像素复现成功。上一轮失去的 45,777 个纹素补回 44,144，仍有 1,633 未恢复；相对原始质量选源净增 69,501。

全部有效来源属于 TRAIN 且在固定观测中具有正分表面证据，Viewer/历史产物保护哈希不变。仅本地 CPU，未发布候选。旧缓存兼容和缺少回退像素 heldout/视觉验收的限制仍保留。完整记录位于 `output/offline_iterations/20261002_nineclip_fallback_2048/REPORT.md`、`comparison_report.json`、`execution_manifest.json`。

## 顺序验证：2026-10-02

`audit_texture_heldout.py` 已实际运行：冻结几何、相机、两版 atlas 与 TRAIN 曝光增益，对九段各 10 个留出时刻进行 UV 纹素中心投影，真人/镜中分别记录，共 178 个有效 frame/view 行。新增 73,072 和未知 56,986 纹素全部评估；旧区域按 UV 线性次序 stride64 抽取 58,421 个纹素，不能把它称为全量旧区域像素审计。来源帧必须属于 TRAIN，所有留出时刻均被排除；老缓存的生成绑定限制仍保留。

| 留出视图与区域 | 误差中位数 / BGR MAE 0–255 | P95 | 留出中至少一次可见纹素 |
|---|---:|---:|---:|
| 真人旧区域（抽样） | 7.3333 | 42.7028 | 48,260 / 58,421 |
| 真人新增区域 | 11.8833 | 62.2095 | 59,993 / 73,072 |
| 镜中旧区域（抽样） | 5.1439 | 42.6081 | 37,866 / 58,421 |
| 镜中新增区域 | 12.6667 | 72.5501 | 43,739 / 73,072 |

两个区域的表面组成不同，这不是同一表面上的前后回归比较，不能据此宣称回退恶化了原区域。原 RGBA 全部保持一致。真人/镜中还分别看到 10,834 / 6,023 个仍未知纹素，说明有限 TRAIN 次选源搜索未覆盖所有留出可见表面；可见性依赖估计几何/遮罩，尚非独立真实表面真值。各帧/纹素样本相关，没有校准验收阈值。报告及最大误差帧的原视频标记图位于 `output/offline_iterations/20261002_sequential_validation/heldout/`。

随后使用独立预览对照 48.43、49.23、50.03，同一片内只切换纹理，保留 native `mesh_local.bin`，未混入 v9 refit mesh。正背左右、放大视图和播放采样能渲染；衣物边缘、领口、手部仍有斑块/错位，基线中也存在。48.43 播放到 249/249；三个定时截图包含播放中画面，但不是逐帧闪烁验收。截图与局限写入 `visual/review.json`。独立预览（未验收）：

- 48.43：`http://127.0.0.1:18769/datasets/f01a3c65542444a1b3dcf62673480872/result/viewer.html`
- 49.23：`http://127.0.0.1:18769/datasets/82b6fd45495a4a53b7928802389e6928/result/viewer.html`
- 50.03：`http://127.0.0.1:18769/datasets/975e32db8f5f4971a399bf7dbf290054/result/viewer.html`

第三步使用 `audit_racket_evidence_boundaries.py` 对 v9/v10 冻结输入生成逐帧投影误差、训练点平均权重、SO(3) 步进/加速度和原视频接触表。zero180–183 真人点缺失；镜中平均点权重 zero181→182 为约 .065→.216，zero180 镜中拍头/拍尖残差约 68.5/60.3 canonical px。zero105 和 zero115 也有点集缺失/恢复。原视频表中存在自动点与可见拍框不一致或运动模糊的嫌疑，需要进一步复核点的角色/几何，不能将自动点当真值，也不能从相关性认定门控导致抖动。v10 的三处加速度相对 v9 基本不变。独立保留帧 P95 重投影计算与原报告差异均 <1e-5 canonical px。

第四步结论：**验证执行完成，验收未完成，不替换主 Viewer**。九段 atlas 未重生成，云 GPU 连接/环境/空闲进程/锁检查通过（RTX 4090 D），本轮没有部署或调度新 GPU job。新的两个审计各 3 项测试通过；主 Viewer 与历史产物保护哈希全部保持一致。完整运行记录为 `output/offline_iterations/20261002_sequential_validation/validation_manifest.json`。

Graphify 默认 Homebrew Python 缺少模块；找到已有 Python.org 安装后执行更新，没有额外安装。根目录扫描包含生成包，运行较久，已停止；改为 Git 管理及非忽略的新源码范围重建，得到 2,930 节点/5,325 边/120 社区，路径重定位回仓库。`graphify-out/source_scope.json` 记录所选文件，图不包含忽略的实验产物或依赖包。

复现第一步（新输出路径，不覆盖本轮）：

```sh
python3 -B viewer/video_import/audit_texture_heldout.py \
  --batch output/multivideo_texture/batch_manifest.json \
  --group-manifest output/offline_iterations/20261001_texture_v2/nine_training_manifest.json \
  --layout output/multivideo_texture/mhr_uv.npz \
  --primary output/offline_iterations/20261002_nineclip_fallback_2048/primary \
  --candidate output/offline_iterations/20261002_nineclip_fallback_2048/fallback \
  --output <new_heldout_audit_directory> --old-stride 64
```

复现第三步，`--centers` 是零起始帧；v10 将 `--run` 中 v9 换为 v10：

```sh
python3 -B viewer/video_import/audit_racket_evidence_boundaries.py \
  --run output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/gpu-fit \
  --video output/video_library/85ade7a072984579831f5cb76e8e5fd3/source.mp4 \
  --centers 105 115 182 --radius 3 --output <new_racket_audit_directory>
```


## 2026-10-02：九段 Viewer 遍历与原生对照

剩余八段使用各自已完成的原生网格和视频，不重新推理；新建独立九段集合，原 v9 主 Viewer 保留。入口：

http://127.0.0.1:18769/datasets/c15130c2b66b58a69696d6d617888501/result/viewer.html

提供上一段/下一段、视频选择、0.25–2×速度、逐帧、重置视角、主选源/像素回退/灰模对照。视频库按九段遍历命名。确定性集合 ID 可重复发布，每段 mesh_meta.video_sha256 校验对应视频，网格、视频和原生材料使用硬链接减少磁盘重复。集合不包含其余八段的联合人体/球拍 refit，不能把共享纹理效果当作原生模型的输出质量。

### 覆盖率

共享 atlas UV 像素覆盖 **98.5271%**，主选源 **96.6384%**，共享训练观测面覆盖 **95.3951%**。以下为每段训练观测面覆盖，按模板面数统计；不是单段生成 atlas 的像素覆盖，不能与共享 UV 指标相减。

| 视频 | 本段训练观测面覆盖 |
|---|---:|
| 48.43 | 88.32% |
| 48.53 | 87.68% |
| 49.03 | 89.79% |
| 49.13 | 90.24% |
| 49.23 | 91.45% |
| 49.33 | 91.23% |
| 49.43 | 86.27% |
| 49.53 | 88.03% |
| 50.03 | 91.27% |

原生单次输出对照入口 `native_viewer.html` 默认且限定原始姿态，关闭跨帧补色、边缘修补和镜中补色；当前帧投影仍使用遮罩/深度可见性。页面实时报告当前视角可见像素的真人/未知覆盖率，分母与 UV 指标不同。49.03 首帧初步正面观察约90%、背面约1.8%（当时边缘修补仍为默认开启；最终对照已关闭，不能将该样例值当作最终固定基准）。背面大面积灰色说明当前原生单帧推理+投影不能提供高质量全角度外观。九段共享合成也是离线外观处理；可让用户一次导入直接看到已生成材料，但不能声称模型一次推理即有98.53%完整纹理。

推荐将已验证的纹理生成流程接到自动导入流水线，保留质量门槛和候选标记；进一步改善边缘、手部和动态质量仍需对几何/遮罩/来源一致性逐项验证。本轮未新增GPU训练。

### 验证

9段的网格长度、视频/网格身份、共享纹理哈希、每段7个主要HTTP资源通过。Browser 验证切换、逐帧、速度、播放、原生对照及当前视角覆盖率；导入服务8项测试通过，两个页面脚本语法检查通过。没有把全部9段完整播放的视觉质量验收作为通过项。Graphify按Git源代码范围重建，排除生成输出。验证记录在 `output/offline_iterations/20261002_viewer_collection/`。

复现：

```sh
python3 -B viewer/video_import/publish_texture_collection.py \
  --batch output/multivideo_texture \
  --fusion output/offline_iterations/20261002_nineclip_fallback_2048/fallback \
  --baseline output/offline_iterations/20261002_nineclip_fallback_2048/primary \
  --layout output/multivideo_texture/mhr_uv.npz \
  --library output/video_library
```
