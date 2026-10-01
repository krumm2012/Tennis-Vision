# 多 Agent 云 GPU 与纹理增量优化：2026-10-01

## 本轮范围与分工

用户选择继续现有 9 段质量优化。复用已完成的 SAM3D/SAM2 结果，共 2,242 真人帧、2,159 镜中有效帧。未重新推理全部视频，未完成新的全身/握拍重拟合。本轮有一次真实云 CUDA 原生回放审计，纹理优化与独立输入审计在本地运行。

| Agent | 责任与文件 | 执行位置 |
| --- | --- | --- |
| gpu_native_audit | `audit_native_batch.py`：回放、拓扑、有限值、缺帧和模型/来源哈希 | 云 RTX 4090，统一 GPU 锁 |
| texture_quality | `texture_consistency.py`、`multivideo_texture.py`：训练选源和固定留出对照 | 本地 CPU |
| evidence_quality | `audit_multivideo_evidence.py`：原片时间轴、SAM2、主体/镜中投影 | 本地 CPU |
| 主线程 | 对照预览、发布来源校验、集成测试、记录与 Git | 本地 |

多个 agent 并行处理独立工作，一张 GPU 的计算通过唯一执行通道及已有 `gpu.lock` 串行。所有权分开；完成子任务后主线程汇总，既有 Viewer、原始档案和人工标注不作为候选输出位置。

## 云端执行组件与真实记录

复用已安装的 SAM/MHR 环境：`/root/tennis-sam3d/venv/bin/python`，Torch 2.4.0+cu121 / CUDA 12.1 / RTX 4090。模型组件和首次生成部署见 [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md) 与 [MULTIVIEW_GPU_ITERATION.md](MULTIVIEW_GPU_ITERATION.md)。本轮仅把独立审计脚本及请求上传到 `audits/<id>/`，没有重新部署共享 Worker 或下载模型。

审计 ID `c3d636e734ab4060ae7bb649ced2281e`。执行前核对 9 个 job 均 ready、GPU 无活动计算；请求将各 job 与原片/归一化片/档案 hash 绑定。脚本获取同一 GPU 文件锁，加载 MHR rig，对原生模型参数 204、形状 45、表情 72 维按批次回放；官方坐标 cm→m、x,-y,-z，镜中再恢复相机 x，保持解剖 ID。缺帧保留零占位并排除回放。

实际远端命令（已有本轮 request 和代码目录）：

```sh
/root/tennis-sam3d/venv/bin/python \
  /root/tennis-viewer/audits/c3d636e734ab4060ae7bb649ced2281e/audit_native_batch.py \
  --root /root/tennis-viewer \
  --request /root/tennis-viewer/audits/c3d636e734ab4060ae7bb649ced2281e/request.json \
  --output /root/tennis-viewer/audits/c3d636e734ab4060ae7bb649ced2281e/reports
```

本地私有连接配置仍为 `deploy/3dpose/cloud_gpu/host.local.json`，不提交凭证。每次新的审计使用新 ID、新请求；先核对对应的远端推理任务和模型 hash，再上传审计脚本。报告、执行日志和下载原件在 `output/multiagent_gpu/native_audit/`，`summary.json` 记录下载文件哈希，`postflight.json` 记录进程释放。

9/9 真人与镜中回放通过原有 1e-4 m 门槛，最大网格欧氏误差 1.01961405e-06 m；无非有限参数/源输出，faces 与实际 rig 逐项一致。保留 83 个镜中缺帧，没有以遮罩存在补造几何。实际 CUDA 运行 42.63 秒，峰值已分配显存 2977 MiB，结束后 GPU 释放。**回放通过只验证保存参数可复现已保存的网格，不是现实三维精度。**

## 独立输入证据

9 段原片、归一化片、档案和显示网格来源链均通过。2,242 帧归一化时间步 0.04 秒，原片 PTS 映射单调；纹理索引始终以归一化视频为准。显示平滑相对原生增加约 1.06–1.56 px 的 P95 投影偏移，因此取色仍用原生网格。

每 5 帧审计，共 900 个视图槽，17 个镜中槽缺原生观察并明确保留。真人/镜中网格轮廓对 SAM2 平均 IoU 0.834 / 0.783。61 个镜中样本与前景真人遮罩重叠超过 5%，可能是遮挡或局部投影偏差，不能据此宣布身份错选。衣物厚度、滑动、阴影和全框清晰度中的背景高频，仍会导致颜色采样不一致。

审计信息在 `output/multiagent_gpu/evidence_quality/quality_inputs.json`、`view_quality.json`、`QUALITY_REPORT.md`，固定 54 槽关键帧拼图 `projection_contact_sheet.jpg`。这些是质量证据，不是人工真值；本轮没有用它们事后删掉纹理留出样本。

## 纹理增量方法与公平对照

各视频训练观测独立取稳健中位色，再按视频等权形成颜色共识。共识仅作选源排序参考，偏离共识的观测软降分，最小乘数 0.05；所有有效面支持保持。邻面选源 Potts 权重 0.25→0.6。输出颜色仍是实际原分辨率视频单一来源像素，没有写入中位色或生成未知内容。

保持旧相机、遮罩、深度规则、曝光 gain、采样帧和留出帧；1,463,590 个固定 `(clip, frame, view, face)` tuple 和 35,176 个有效面，评价分母完全一致。旧观测源码 hash 与本轮选源源码 hash 分开记录，源码快照随结果保存。

| 固定评价项 | 上轮 9 段 | 本轮候选 |
| --- | ---: | ---: |
| 留出颜色 MAE 平均 / 255 | 19.716 | 12.832 |
| 留出颜色 MAE P95 / 255 | 61.363 | 44.508 |
| 邻面颜色差平均 / 255 | 9.542 | 5.356 |
| 邻面来源标签切换 | 27.03% | 36.89% |
| 有效 UV 覆盖 | 96.7307% | 96.6384% |

留出平均颜色残差下降 34.9%，9/9 片改善；邻面颜色差下降 43.9%。标签切换增加，不能称选源更连续；该指标描述静态 atlas 的相邻三角形来源，并不是播放时逐帧换纹理。覆盖净下降 0.0923 百分点，旧有效 texel 丢失 45,777、新增 42,206，共同有效 3,696,680。邻面颜色差是接缝代理，也包含真实衣物边界，不等于几何或新视角准确率。

产物在 `output/multiagent_gpu/texture_quality/consistency_v1/`：PNG、逐 texel 来源、UV 网格、标准报告、公平比较、旧版本备份、源码快照。`candidate_integrity.json` 验证全部 3,738,886 个有效 texel 与固定观测选源一致。未知 texel 透明，未填灰色表面。

## 复现与本地预览

```sh
python3 -B viewer/video_import/audit_multivideo_evidence.py \
  --batch output/multivideo_texture \
  --output output/multiagent_gpu/evidence_quality --step 5

python3 -B viewer/video_import/texture_consistency.py \
  --observations output/multivideo_texture/fusion_independent \
  --batch output/multivideo_texture \
  --layout output/multivideo_texture/mhr_uv.npz \
  --output output/multiagent_gpu/texture_quality/consistency_v1

python3 -B viewer/video_import/publish_texture_review.py \
  --batch output/multivideo_texture \
  --fusion output/multiagent_gpu/texture_quality/consistency_v1 \
  --baseline output/multivideo_texture/fusion_independent \
  --layout output/multivideo_texture/mhr_uv.npz \
  --destination output/video_library/5242a81d6773428090c4ed5274f3015b
```

[本轮纹理对照 Viewer](http://127.0.0.1:18769/datasets/5242a81d6773428090c4ed5274f3015b/result/viewer.html) 支持合成候选、上轮 9 段、灰模。发布前同时校验新旧 PNG hash 和 UV 布局；对照标为“上轮合成”，避免误标为单视频。浏览器已检查正背面、侧面、三个模式和 249/249 末帧；截图在 `output/multiagent_gpu/`。白衣及肤色更连续，但手部、衣物边界和未观测孔洞仍存在。候选独立发布，主数据集球拍和标注保持原位置。

39 项 Python 集成测试、球拍 root-invariant 插值测试、HTML 脚本语法检查通过。错误 UV 对照会在创建发布目录前被拒绝。按 AGENTS 运行 graphify 更新仍因本机缺少模块失败，未额外安装。

## 后续重点

2026-10-01 后续已补自动 MHR 分项进度台账、历史失败/候选回填及九段视频的 511 子集分析，见 [MHR_FIT_PROGRESS.md](MHR_FIT_PROGRESS.md)。精选四段能保留九段约 99% 的观察支持，但共享身份、相机/尺度与真实手部接触仍待完成，不能据此记成完全拟合。

1. 使用 SAM2 内部局部清晰度和有效投影采样分辨率，减少球网/背景对选帧评分的影响；建立新固定对照协议后再改观测门槛。
2. 对手部/肩颈/衣物腰线补可靠对应，做局部几何及遮挡约束。原生回放正确不能替代这一步。
3. 握拍继续使用 [tennis-racket-fitting skill](../../skills/tennis-racket-fitting/SKILL.md)；本轮没有完成真实握柄棱位、网格表面接触或通过全身重拟合验收。
