# 3D 网球人体、球拍与纹理：可复现操作手册

记录日期：2026-10-01（Asia/Shanghai）。仓库：`/Users/krum5539/Documents/Tennis-Vision`。

## 0. 从哪条路径开始

| 目标 | 执行章节 | 是否运行 GPU |
| --- | --- | --- |
| 复算现有九段的颜色对照与数量收益 | 1、2、6、7、10 | 否，本地 CPU |
| 为新素材生成真人/镜中人体与遮罩 | 1–5、6、10 | 是，SAM3D/SAM2 |
| 为已有素材重新拟合握拍和 MHR 姿态 | 1、2、8、9、10 | 球拍后处理本机 MPS；MHR 优化云 CUDA |
| 只恢复中断任务的结果 | 9 | 下载不重启计算；原任务可能仍在运行 |

本手册记录已实现流程及实际失败案例。共享身份 shape/scale 优化、实测相机/尺度、真实手部表面接触和握柄棱位仍未验收完成；不要把教程中运行命令成功理解为这些工作已经完成。分项完成度见 [MHR_FIT_PROGRESS.md](MHR_FIT_PROGRESS.md)。

复现分三层：①保存原始数据/结果并验证 SHA，可以精确取回历史产物；②在相同输入、代码与环境上复算评价；③重新做 GPU 推理/优化，必须重新报告实际结果。GPU 算子、解码器和平台差异可能改变数值，本项目尚未建立跨硬件逐字节确定性协议，不承诺新推理 PNG/NPZ 与旧文件 SHA 相同。

## 1. 固定代码和独立输出目录

以下命令默认在 Mac 仓库根目录执行；云命令单独标记。各章变量在同一个终端会话中使用。每轮修改 `round_001`，不要重用已完成输出。

```sh
cd /Users/krum5539/Documents/Tennis-Vision
REPO="$PWD"
GPU_CONFIG="$REPO/deploy/3dpose/cloud_gpu/host.local.json"
RUN="$REPO/output/reproduction/round_001"
BATCH="$REPO/output/multivideo_texture"
LAYOUT="$BATCH/mhr_uv.npz"
OBS="$BATCH/fusion_independent"
mkdir -p "$REPO/output/reproduction"
mkdir "$RUN"
git rev-parse HEAD > "$RUN/local_git_commit.txt"
git status --short > "$RUN/local_git_status.txt"
```

已核对的功能基准：`b84045a`，其前一提交 `9de5610` 引入 MHR 台账与视频子集分析。后续文档提交不替代历史 GPU 版本：初段云生成基准为 `d920d1e…`，其余为 `dabccb9…`，每段实际文件版本以 `remote_manifest.json/code_sha256` 为准。恢复旧版本时使用独立 checkout/worktree，保留当前工作区及用户标注。

输出目录约定：

```text
output/multivideo_texture/
  inventory.json / batch_manifest.json / source_index.json
  mhr_uv.npz / mhr_uv.json
  clips/<stem>/
    original.video / source.mp4 / record.json
    attempts/0001/work/
      remote_job.json / request.json / remote_manifest.json / worker.log
      reconstruction.npz / multiview_manifest.json
    result/mesh_meta.json / mirror_*.json / person_masks_sam2.png / viewer.html
  fusion_independent/   # 旧九段基准 + 固定训练/留出观测缓存
output/multiagent_gpu/
  native_audit/         # 云回放请求、脚本、日志与报告
  texture_quality/consistency_v1/  # 已保存颜色候选、来源、报告、源码快照
output/mhr_fit_progress/ledger.json / events/<sha>.json
output/reproduction/<round>/     # 新复算、新拟合与环境快照
```

`output/` 不进 Git。Git 只保存方法和代码，不能单靠 `git clone` 恢复视频、权重、人工标注或历史拟合；备份方法见第 10 节。

## 2. 环境、模型与云组件

### 2.1 本次实测环境

| 位置 | Python / 主要依赖 | 用途 |
| --- | --- | --- |
| Mac | Python 版本/路径见快照；Torch 2.10.0，NumPy 2.4.2，SciPy 1.17.0，Numba 0.63.1，OpenCV 4.13.0.92，Ultralytics 8.4.14 | 服务、包装、MPS 观测与球拍拟合、CPU 纹理 |
| 云 GPU | Python 3.12.9，Torch 2.4.0+cu121，NumPy 1.26.4，OpenCV 4.11.0.86，Ultralytics 8.4.7 | SAM3D、SAM2、MHR CUDA |
| GPU | RTX 4090，24 GB；本次 driver 610.43.03 | 共用 `gpu.lock`，串行计算 |
| Viewer | Three.js vendored `three-0.180.0.min.js`；本地 Docker nginx | 18769 静态显示，18768 Mac API |

实际环境和代码清单已只读采集到 `output/reproduction/2026-10-01/`：`local_environment.json`、`cloud_environment.json`、`local_packages.txt`、`cloud_packages.txt`、`local_code_hashes.json`。可随 Git 保存的摘要/证据 hash 在 [REPRODUCTION_BASELINE.json](REPRODUCTION_BASELINE.json)。包清单记录已安装 distribution 版本，不含安装源 URL、editable 路径和 CUDA 编译选项，**不是已验证的新机器 lockfile**。

后续每轮重新记录当前运行 Python 与版本；不要从另一个环境的 `pip` 获取清单：

```sh
python3 -m pip list --format=json > "$RUN/local_packages.json"
python3 -c 'import sys,platform,json;print(json.dumps({"python":sys.version,"executable":sys.executable,"platform":platform.platform()}))' > "$RUN/local_runtime.json"
ffmpeg -version > "$RUN/ffmpeg_version.txt"
export RUN
python3 - <<'PY'
import os,sys,json
from pathlib import Path
sys.path.insert(0,'viewer/video_import')
from cloud_adapter import config,ssh,copy
c=config('deploy/3dpose/cloud_gpu/host.local.json');out=Path(os.environ['RUN'])
code='import sys,json,platform,importlib.metadata as m;print(json.dumps({"python":sys.version,"platform":platform.platform(),"packages":{d.metadata["Name"]:d.version for d in m.distributions() if d.metadata["Name"]}}))'
(out/'cloud_runtime.json').write_text(ssh(c,[c['python'],'-c',code]))
(out/'gpu_driver.txt').write_text(ssh(c,['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']))
copy(c,c['root']+'/worker_config.json',out/'worker_config.json')
PY
```

Worker 配置只有部署代码/模型路径，SSH 凭证不写入报告；未来配置若扩展了敏感字段，应先只提取需要的版本字段。模型身份逐个按表中 SHA 校验，数据身份以第 5/6 节及生成清单校验。

### 2.2 安装顺序与固定版本

现有云主机使用 SSH 调用 Python，不是 GPU Docker 镜像。`deploy.py` 只上传本项目脚本并检查环境，不安装模型依赖，也不下载权重。

新主机按以下顺序准备：

1. 安装可用 NVIDIA driver、Python 环境及与该环境匹配的 CUDA PyTorch。
2. 准备 SAM3D 源码及其依赖，按官方安装说明处理 Detectron2 等编译包；准备已获访问权限的权重。
3. 安装对应 SAM2 源码及 small checkpoint；核对 SAM2 包的版本约束，不能把现有环境中的 Torch 2.4.0 当成最新安装要求。
4. 安装当前 pipeline 使用的 Ultralytics 人物分割及本地镜中 pose/球拍模型。当前提示模型采用 YOLO，不要求额外安装 SAM3 detector。
5. 用实际文件 hash 验证 checkpoint/模型配置/MHR rig，最后部署本项目 Worker 并运行 `check`。

复现历史环境优先恢复已有环境与安装源码快照。已实测的源码 revision：SAM3D `b5c765a0d89d789985e186d396315e7590887b94`，SAM2 `2b90b9f5ceec907a1c18123530e92e794ad901a4`。仅有 revision 不含本地补丁、编译产物，备份时也应保留实际源码及环境。

当前官方 SAM3D 安装说明提供 Python/PyTorch、依赖和权重访问步骤；新装依此准备，再冻结实际安装结果。[SAM3D 安装说明](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md)

当前 SAM2 上游要求 Python ≥3.10、Torch ≥2.5.1、torchvision ≥0.20.1。历史环境已实际推理通过，但不能据此把它写成上游支持的组合。[SAM2 安装说明](https://github.com/facebookresearch/sam2#installation)

### 2.3 云模型位置与身份

| 组件 | 本轮实际位置 | SHA-256 |
| --- | --- | --- |
| SAM3D checkpoint | `/root/tennis-sam3d/weights/model.ckpt` | `3b1cb897f4bbd977bf81cbb0b30780a9582681ac642ee112865790ceb4d66056` |
| SAM3D config | `/root/tennis-sam3d/weights/model_config.yaml` | `d2e772e108b8727e9367681845fecb32806144acd0debc20868d100689470570` |
| MHR rig | `/root/tennis-sam3d/weights/assets/mhr_model.pt` | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |
| SAM2 small | `/root/tennis-sam3d/sam2_work/sam2.1_hiera_small.pt` | `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` |
| 云人物提示模型 | `/root/tennis-viewer/models/yolo11n-seg.pt` | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |

云 Python `/root/tennis-sam3d/venv/bin/python`；SAM3D 代码 `/root/tennis-sam3d/sam-3d-body`；SAM2 `/root/tennis-sam3d/sam2`；任务根 `/root/tennis-viewer`。本机还需 `yolo26m-pose.pt`（镜中观察）、`yolo26s-seg.pt`（球拍 ROI），以及 `output/sam3d_cloud/wilson_mesh.bin` / `wilson_model.json`。新文件不能默认冒用旧模型或标定。

### 2.4 配置、检查、部署

首次创建本机私有配置；已有 `host.local.json` 不覆盖。模板为 `deploy/3dpose/cloud_gpu/host.example.json`，配置 SSH host/port、远端 root/python 和下列白名单 env：

```json
{
  "SAM3D_BODY_CODE": "/root/tennis-sam3d/sam-3d-body",
  "SAM3D_WEIGHTS": "/root/tennis-sam3d/weights",
  "VIEWER_SEGMENTATION_WEIGHTS": "/root/tennis-viewer/models/yolo11n-seg.pt",
  "VIEWER_MULTIVIEW": "1",
  "VIEWER_SAM2_WEIGHTS": "/root/tennis-sam3d/sam2_work/sam2.1_hiera_small.pt"
}
```

下面是**只读检查**，不部署/不推理。`check` 成功需 cuda=true、missing 为空；活动进程未释放时先等既有任务完成。

```sh
python3 - <<'PY'
import sys,json
sys.path.insert(0,'viewer/video_import')
from cloud_adapter import config,ssh
c=config('deploy/3dpose/cloud_gpu/host.local.json')
print(ssh(c,['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']))
print(ssh(c,[c['python'],c['root']+'/code/worker.py','check','--root',c['root']]))
PY
```

需要新部署时，确认 GPU 无正在使用共享 `code/` 的任务后执行：

```sh
python3 -B deploy/3dpose/cloud_gpu/deploy.py --config "$GPU_CONFIG"
```

保留部署输出和云 `worker_config.json`；其中 Git revision 与实际上传文件 SHA 分别记录。部署不删除历史任务或权重，运行中重部署会改变未来导入的文件版本，因此不能用于复现旧任务。

## 3. 输入视频、时间轴与处理契约

本轮输入：`/Users/krum5539/Desktop/Camera/2026-09-30`，九段 48.43–50.03。实际画面 2560×1440，原片容器有 50/1 名义字段但平均约 25 fps；不要只根据 r_frame_rate 声称有效 50 fps。

归一化由导入/批处理脚本完成：取首个视频流、去音频、最大宽 2560、偶数尺寸、25 fps、H.264/yuv420p/faststart。原片保留 `original.video`，归一化 `source.mp4` 才是所有模型、标注与纹理的时间轴。零起始 frame i 对应约 i/25 秒，Viewer 通常显示 i+1。现有归一化共 2,242 帧，48.53 为 250，其他八段各 249；不要补成统一 250 帧。

可单独审查原片，不修改它：

```sh
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration \
  -of json /Users/krum5539/Desktop/Camera/2026-09-30/48.43.mp4
```

SAM 网络内部按模型分辨率缩放；2560×1440 是输入/取色分辨率，不代表所有神经网络全程以原生像素分析。采集更高 fps 时必须同步修改导入时间轴才能保留快挥拍信息。

NPZ：禁用 pickle；真人 vertices `[F,18439,3]`、faces `[36874,3]`、roots `[F,3]`、focal `[F]`、joints `[F,70,3]`；姿态 `[F,204]`、身份 `[F,45]`、表情 `[F,72]`。米、相机轴 x,-y,-z；`vertices + source_roots` 才是相机空间位置。镜中保存独立 `mirror_*`、`mirror_valid` 和 SAM2 confidence；有效帧之外保留缺失。

## 4. 新视频的真实云生成与拉回

### 4.1 九段批处理

**此命令会使用云 GPU**。新源目录使用新的输出批次，不复用 `output/multivideo_texture` 作为不同文件的存放目录。当前脚本匹配一级 `*.mp4`，不递归；同名文件内容改变会拒绝。批处理会在 Mac 使用 MPS 执行镜中 pose 模型，当前没有批处理 `--device cpu` 参数。

恢复现有九段时沿用第 1 节 `$BATCH` 和以下源路径。处理新素材时先把源目录改为实际新目录，设置 `BATCH="$RUN/new_batch"`、`LAYOUT="$RUN/mhr_uv.npz"`、`OBS="$RUN/fusion_independent"`；随后第 5 节导出/索引也使用这些新路径。

```sh
python3 -B viewer/video_import/batch_multivideo.py \
  --source-dir /Users/krum5539/Desktop/Camera/2026-09-30 \
  --output "$BATCH" --config "$GPU_CONFIG" \
  --mirror-model /Users/krum5539/Downloads/yolo26m-pose.pt
```

现有批次重跑会先验证已有 NPZ/清单 hash，完整结果不会重复推理；不完整结果按原 job 恢复拉取。曾用以下 seed 参数复用同源 48.43；只在原片 SHA 完全相同时成立，新人物/新视频省略：

```text
--seed-dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3
--seed-native output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native
```

批处理内部记录失败后继续下一段，因此**程序退出码不足以证明全部成功**。运行后检查：

```sh
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('output/multivideo_texture/batch_manifest.json').read_text())
assert r['status']=='ready' and r['clips'] and all(c['status']=='ready' for c in r['clips'])
print('all ready:',len(r['clips']))
PY
```

新批次替换上述检查路径。镜面也要单独检查 `mirror_calibration_report.json` 和 `mesh_meta/mirror_available`；ready 不能代替镜面验收。

### 4.2 单视频导入并集成球拍

以下启动 Mac API（已有同端口服务时不重复启动）：

```sh
python3 -B deploy/3dpose/cloud_gpu/start_local.py \
  --config "$GPU_CONFIG" --port 18768 \
  --racket-model /Users/krum5539/Downloads/yolo26s-seg.pt \
  --mirror-pose-model /Users/krum5539/Downloads/yolo26m-pose.pt
```

在 `http://127.0.0.1:18768/import.html` 上传新视频并生成；默认视频仍用 `/default/viewer.html`。单次调用 `cloud_adapter.py --video ... --output ... --config ...` 只返回重建 NPZ，不自动集成球拍或发布页面。其 `--base-job` 仅复用同一归一化视频的真人推理，不能传不同视频任务。

### 4.3 推理方法与结果验收

1. 真人选框跟踪 → SAM3D → 原生 MHR 参数/网格/相机。
2. 镜中独立人物：全图水平翻转后 SAM3D，输出网格/关节/root 的 x 和 2D x 恢复一次。MHR 解剖 ID 保持；COCO 镜中检测匹配另有左右标签交换。
3. YOLO 框作提示，SAM2 双对象视频传播生成置信度遮罩；真实与镜中像素冲突消解。不能将 YOLO 多边形当成 SAM2 结果。
4. Worker 持有共用 `gpu.lock`；多 agent 可并行整理证据，GPU 计算仍串行。
5. 下载 `.part` → 源视频与档案 hash 一致后更名 → `package_result.package` 校验帧数/拓扑/有限值 → 独立 result。
6. 原生网格用于投影取色；平滑网格用于显示，避免显示稳定造成取色漂移。R 为真人、G 为镜中 SAM2；镜中图集只有通过反射关联后启用。

当前九段真人 2,242、镜中有效 2,159，镜中 83 帧缺失保留；非空 mask 不能证明同帧镜中 3D 有效。完整 job 表在 [MULTIVIDEO_TEXTURE_ITERATION.md](MULTIVIDEO_TEXTURE_ITERATION.md)。

## 5. UV 导出与视频来源索引

UV 必须由**同一 MHR rig**导出；不能套另一拓扑的网上模板。已有 `mhr_uv.npz/json` 直接保留，旧 NPZ SHA 为 `a79adc6692464cc971fc9f8d28cfca3db0f3506a0bcfffe1605ed19cc2e5364c`。

新导出时，上传 `export_mhr_uv.py` 到私有新目录，在云 Python 上执行，然后下载 NPZ/JSON；它不做人体推理。示例调用方法：

```python
# 从仓库根目录的 Python 脚本执行；export_dir 使用新的绝对远端目录。
from pathlib import Path
import sys
sys.path.insert(0,'viewer/video_import')
from cloud_adapter import config,ssh,copy
c=config('deploy/3dpose/cloud_gpu/host.local.json')
export_dir=c['root']+'/exports/uv-round-001'
ssh(c,['mkdir','-p',export_dir])
copy(c,Path('viewer/video_import/export_mhr_uv.py'),export_dir+'/export_mhr_uv.py',upload=True)
copy(c,Path('viewer/video_import/run_records.py'),export_dir+'/run_records.py',upload=True)
ssh(c,[c['python'],export_dir+'/export_mhr_uv.py','--model',c['env']['SAM3D_WEIGHTS']+'/assets/mhr_model.pt','--output',export_dir+'/mhr_uv.npz'])
destination=Path('output/reproduction/round_001')
copy(c,export_dir+'/mhr_uv.npz',destination/'mhr_uv.npz')
copy(c,export_dir+'/mhr_uv.json',destination/'mhr_uv.json')
```

新批次需要建立 `source_index.json`，旧批次已有的索引原件不能被“补写”覆盖。以下在**新批次**执行，替换 batch 路径；脚本拒绝覆盖已有索引，并验证源/档案与 manifest：

```sh
export BATCH
python3 - <<'PY'
import os,json,hashlib
from pathlib import Path
import numpy as np
batch=Path(os.environ['BATCH'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((batch/'batch_manifest.json').read_text())
assert manifest['status']=='ready'
rows=[]
for row in manifest['clips']:
 folder=Path(row['folder']);work=folder/'attempts/0001/work'
 meta=json.loads((folder/'result/mesh_meta.json').read_text())
 remote=json.loads((work/'remote_manifest.json').read_text())
 video_sha=sha(folder/'source.mp4');archive_sha=sha(work/'reconstruction.npz')
 assert video_sha==meta['video_sha256']==remote['source_sha256']
 assert archive_sha==remote['artifact_sha256']['reconstruction.npz']
 with np.load(work/'reconstruction.npz',allow_pickle=False) as d:
  assert len(d['vertices'])==meta['frames']
  count=int(d['mirror_valid'].sum()) if 'mirror_valid' in d else 0
 rows.append({'video':row['video'],'folder':str(folder.resolve()),'original_sha256':sha(folder/'original.video'),
  'normalized_sha256':video_sha,'archive_sha256':archive_sha,'remote_job':row['remote_job'],
  'remote_code_revision':remote.get('git_commit'),'frames':meta['frames'],'fps':meta['fps'],
  'resolution':meta['image_size'],'mirror_sam3d_valid_frames':count})
with (batch/'source_index.json').open('x') as f:json.dump({'clips':rows},f,indent=2)
PY
```

## 6. 真实视频纹理合成与颜色对照

### 6.1 方法和固定参数

不是逐帧手修。每 5 帧抽样、每 25 帧留出；训练候选采用正深度、SAM2 内部高置信度、深度可见性、较清晰且非掠射表面。当前中心/像素 mask 门槛 >0.92、深度差 <0.025 m、面朝向幅值 >0.3；可见性深度图宽 1280，颜色来自原分辨率视频。确切参数以保存的源码 SHA 为准。

每个 UV 像素通过 MHR UV 三角形的重心坐标找到人体表面，投影到选中的 clip/frame/view 取实际像素。`independent` 模式使用镜中有效原生网格；x 已还原，不再翻转。曝光校正、邻面选源用于减少接缝，未知区域 alpha=0；RGB 边缘扩展不增加已观测覆盖。

一致性候选：每片训练色先取稳健中位数，再视频等权形成参考；色差软降分尺度 24/255，最低权重 0.05；邻面 Potts 0.25→0.6、5 次迭代。中位色只决定来源排序，不直接画进 atlas。留出颜色不参与选源。

### 6.2 先复算固定评价（本轮已实跑）

这一节仅用于已有完整观测和基准的批次。新素材直接从 6.3 创建基准；完整一致性合成也会同时生成固定对照报告。

```sh
python3 -B viewer/video_import/texture_consistency.py \
  --observations "$OBS" --batch "$BATCH" --layout "$LAYOUT" \
  --output "$RUN/color_check" --compare-only > "$RUN/color_check.log"
```

`--compare-only` 仍复制基准产物并保存选源数组/留出 tuple；不会生成新候选 atlas。输出 `comparison_report.json`，应使用同一 1,463,590 个基准支持 tuple 比较。历史结果：平均 MAE 19.716→12.832、邻面色差 9.542→5.356；标签切换 27.03→36.89% 是退化，不能只报颜色改善。

### 6.3 合成新批次 / 复用旧观测

新批次完成镜面验收和 UV 后先生成质量基准：

```sh
python3 -B viewer/video_import/multivideo_texture.py \
  --batch "$BATCH" --layout "$LAYOUT" --output "$RUN/fusion_independent" \
  --size 2048 --step 5 --mirror-source independent
```

此处 `$BATCH` 对应当前要研究的批次。旧批次严格固定观测对照则使用已有 `$OBS`，不要重建缓存；源码 hash 变化会让观测缓存被重算。完整一致性合成（本机 CPU）执行：

```sh
python3 -B viewer/video_import/texture_consistency.py \
  --observations "$OBS" --batch "$BATCH" --layout "$LAYOUT" \
  --output "$RUN/consistency" --size 2048 > "$RUN/consistency.log"
```

新批次把 `$OBS` 指向它刚生成的质量基准。输出 `body_texture_rgba.png`、`texture_sources.npz`、`appearance_mesh.npz`、`texture_report.json`、固定比较和基准副本。UV 面支持、实际 alpha 覆盖、留出色差分别报告，不等于三维精度。

旧九段实际 UV 覆盖 96.7307%，一致性候选 96.6384%。候选 PNG SHA 为 `70b9a9afdf1e65e244dc702e89354efcc78e3834b8891725dab1534f67edc092`；旧 PNG 为 `fde36a1283ca8a35adcd2bcbff1b1097e18a70c9a4a051379e622bf17e6d1192`。保存的两份源码在 `consistency_v1/*_snapshot.py`；基准观测源码身份为 `b270ced…`，不能用最终代码版本冒充其生成代码。

### 6.4 独立发布 Viewer

```sh
REVIEW_ID=$(python3 -c 'import uuid;print(uuid.uuid4().hex)')
python3 -B viewer/video_import/publish_texture_review.py \
  --batch "$BATCH" --fusion "$RUN/consistency" --baseline "$OBS" \
  --layout "$LAYOUT" --destination "$REPO/output/video_library/$REVIEW_ID"
```

脚本校验新旧纹理与 UV hash，拒绝替换非纹理评审 dataset。纹理评审播放首段姿态，并不自动集成球拍。历史对照入口：`http://127.0.0.1:18769/datasets/5242a81d6773428090c4ed5274f3015b/result/viewer.html`。

## 7. 复算需要几段视频

```sh
python3 -B viewer/video_import/analyze_video_sufficiency.py \
  --observations "$OBS" --layout "$LAYOUT" \
  --source-index "$BATCH/source_index.json" --output "$RUN/video_sufficiency" \
  > "$RUN/video_sufficiency.log"
```

现有九段的 511 个组合：精选 48.53/49.23/49.33/50.03 达到九段面支持 99.258%、UV 面积支持代理 99.670%；95.198% 已支持面至少在五片出现。代理不等于实际 atlas 覆盖，更不是“拟合完成度”。正式正反手标签仍 unknown；不要将固定机位连续九片当九个标定视角。

下一轮初始预算：正手两角度 + 实际反手类型两角度 + 慢转身 + 握拍/拍面近景，共六个目标角色；已验证旧片可复用。完整采集/停止条件见 [MHR_FIT_PROGRESS.md](MHR_FIT_PROGRESS.md)，原抽样和来源在 `output/video_sufficiency/capture_plan/`。

## 8. 握拍、球拍与 MHR 联合优化

### 8.1 私有 staging 与清晰关键帧

球拍后处理沿用 demo 的 MCP/PIP 掌内握点、腕锚点和有向杆轴；真人/镜中 ROI 与拍框关联不能混淆，遮挡柄端不补成观测。先用原视频清晰帧填写 handle_end、tip、rim_side、rim_opposite，拍喉和物理 A/B 有证据才填写。全部标注绑定视频 SHA、尺寸、fps、零起始帧号。

当前主 Viewer `85ade…` 的用户标注和默认球拍不要作为实验写入位置。先复制到独立 dataset staging，`record.json/attempt` 指向与本次 native 档案相同的工作目录。若重新推理相机发生变化，先从新档案重新包装，并重新估计镜面、球拍和手部观测；不能只复制旧 root/标定以绕过门槛。`package_result.package` 目标目录必须不存在。

在已有原生档案、同源 person_candidates 和模型条件满足的 staging 中按顺序执行（这里 `$STAGE` 需设置为实际 staging dataset）：

从已有合格且相机一致的 dataset 创建完全独立的副本。下面以当前批次首段为例；它不包含旧主 Viewer 的人工球拍标注，因此需要在副本中生成球拍观察与校准。`copytree` 默认复制文件，不将可写标注硬链接回原目录。

```sh
SOURCE_DATASET="$BATCH/clips/48.43"
STAGE="$RUN/racket-stage"
export SOURCE_DATASET STAGE
python3 - <<'PY'
import os,shutil
from pathlib import Path
source=Path(os.environ['SOURCE_DATASET']);target=Path(os.environ['STAGE'])
assert (source/'record.json').is_file() and (source/'result/mesh_meta.json').is_file()
shutil.copytree(source,target)  # 已存在则失败，原 source 不修改
PY
```

```sh
python3 -B viewer/video_import/estimate_mirror.py \
  --dataset "$STAGE" --result "$STAGE/result" \
  --model /Users/krum5539/Downloads/yolo26m-pose.pt --device mps
python3 -B viewer/video_import/multiview_constraints.py --dataset "$STAGE" --result "$STAGE/result"
python3 -B viewer/video_import/racket_observations.py \
  --dataset "$STAGE" --result "$STAGE/result" \
  --model /Users/krum5539/Downloads/yolo26s-seg.pt --device mps
python3 -B viewer/video_import/racket_keypoints.py --dataset "$STAGE" --result "$STAGE/result"
python3 -B viewer/video_import/fit_dataset_grip.py --dataset "$STAGE" --result "$STAGE/result"
python3 -B viewer/video_import/racket_review_frames.py --dataset "$STAGE" --result "$STAGE/result"
python3 -B viewer/video_import/audit_racket_motion.py "$STAGE/result/racket_poses.json" --check
python3 -B viewer/video_import/audit_racket_direction.py "$STAGE/result/racket_poses.json" \
  --keypoints "$STAGE/result/racket_keypoints.json" --meta "$STAGE/result/mesh_meta.json"
```

镜中 2D 缓存只有 video/model hash 一致时才能 `--reuse-observations`。球拍方向观测留出与 MHR 自动帧划分各自记录，不能混用分母。规则：短且两端支持的缺口才插值；无支持/长缺口保持隐藏；旋转用 SO(3)，不能靠强平滑宣称方向正确。主流程当前单右手；双手反手需新增职责和接触模型。

### 8.2 复现历史 MHR 实验

历史原生档案、相机一致 repacked 输入均已保留，可以使用它们重做同条件实验；不要读取后来修改的主 Viewer 标注替换历史验证集。

```sh
FIT_DATASET="$REPO/output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/repacked"
NATIVE="$REPO/output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native"
FIT_OUT="$RUN/mhr-fit"
python3 -B viewer/video_import/run_fullbody_refit.py \
  --dataset "$FIT_DATASET" --native-output "$NATIVE" --output "$FIT_OUT" \
  --config "$GPU_CONFIG" --allow-assumed --allow-automatic > "$RUN/mhr-fit.log" 2>&1
```

**这是调用真实云 CUDA 的候选实验**。假设尺寸和自动轮廓尚未验证，所以使用显式临时实验 flags；它们不能让实测尺寸/人工验证阶段自动通过。已有历史运行本轮没有再次调用。

常规准备需六个清晰真人完整关键帧，覆盖时间跨度至少 min(1 秒,半片时长)；存在镜面时还需至少两帧镜中 tip/rim。临时自动实验需至少八个真人清晰帧；每第五个 evidence 条目留出。不可见点不填伪坐标。用户暂无实测，继续 measured=false。

当前优化方法：先用同一 MHR head 重放身份/姿态/表情，验证网格及关节误差；每帧优化全局/骨骼姿态与球拍刚体位姿 120 步，加入人体保真、轮廓/关键点、镜面、掌内接触代理和时序损失。shape、scale、expression、camera roots 保持固定，**尚不是跨片共享身份拟合**。关节圆柱损失尚不是手指网格与柄的真实接触。

门槛：native replay ≤1e-4 m；相机 root 对齐 2 mm、focal 相对 0.1%；人体位移 P95 <0.03 m；固定留出 after <0.9×before。真实报告还需握拍/方向/视觉和完整性验收。求解返回 ready 或 `full_body_joint_fit_completed=true` 只表示完成计算。

历史 run `2dd9765…` 因 root 最大约 4.31 mm 不一致失败；`8ba0f427…` 完成 120 步，掌点 gap 6.45→0.44 mm，但手/柄方向 32.18→34.01°、留出 15.239→15.249 px，数值验收拒绝，未发布。历史目录必须一起备份，包括 repacked；没有保存输入就不能精确复做旧实验。

### 8.3 每轮完成度与验收

`run_fullbody_refit.py` 自动写 `progress_attempt.json`、`completion.json`、总台账和内容寻址事件。执行状态、验收状态、分项门槛和实际源代码 hash 分开；失败/中断也保留。历史或人工下载后补录：

```sh
python3 -B viewer/video_import/mhr_fit_progress.py --output "$FIT_OUT"
```

正、背、左、右、斜视与握柄近景检查分布式清晰帧，截图前确认页面真实帧号；再看原视频/镜中重投影和播放。保存选帧表、截图 hash、接触/穿透、方向、缺失、抖动与留出报告。未过门槛继续保留既有球拍；新视角显示好看不能替代观测证据。

## 9. SSH 中断恢复与原生回放审计

### 9.1 普通 SAM 生成恢复

`cloud_adapter.py` CLI **没有** `--resume-job`；重跑其 `run` 会建新 job。批处理重跑已自动复用已有 remote_job；单视频直接用 `collect`：

```python
import sys,json
from pathlib import Path
sys.path.insert(0,'viewer/video_import')
from cloud_adapter import config,collect
folder=Path('output/multivideo_texture/clips/48.53')
work=folder/'attempts/0001/work'
job=json.loads((work/'remote_job.json').read_text())['job_id']
collect(folder/'source.mp4',work,config('deploy/3dpose/cloud_gpu/host.local.json'),job,poll=10)
```

它等待原任务、校验源/档案 hash 并下载；不会启动新推理。若状态 failed，先读 worker.log，不能用 ready 的其它视频替代。

### 9.2 MHR 拟合恢复

`run_fullbody_refit.py` 再次执行也会建立新尝试，当前没有自动 resume CLI；`mhr_fit_progress.py` 只补记本地证据，**不会下载或启动 GPU**。恢复按原 job 检查、下载、验证、补录。以下脚本用于已有 `$FIT_OUT` 的任务：

```sh
export FIT_OUT
python3 - <<'PY'
import os,sys,json
from pathlib import Path
sys.path.insert(0,'viewer/video_import')
from cloud_adapter import config,ssh,copy
from run_records import sha256
from mhr_fit_progress import record_fit
out=Path(os.environ['FIT_OUT']);c=config('deploy/3dpose/cloud_gpu/host.local.json')
job=json.loads((out/'remote_job.json').read_text())['job_id']
assert len(job)==32 and all(x in '0123456789abcdef' for x in job)
remote=c['root']+'/jobs/'+job
status=json.loads(ssh(c,[c['python'],c['root']+'/code/worker.py','status','--root',c['root'],'--job',job]))['status']
assert status in ['ready','failed','needs_input'], '仍在计算；稍后检查，不新建任务'
copy(c,remote+'/run_manifest.json',out/'remote_manifest.json')
copy(c,remote+'/worker.log',out/'worker.log')
manifest=json.loads((out/'remote_manifest.json').read_text())
request=json.loads((out/'request.json').read_text())
assert manifest['remote_job_id']==job and manifest['source_sha256']==request['source_sha256']
if status=='ready':
 copy(c,remote+'/refit_output/mhr_refit_candidate.npz',out/'mhr_refit_candidate.npz.part')
 assert sha256(out/'mhr_refit_candidate.npz.part')==manifest['artifact_sha256']['mhr_refit_candidate.npz']
 (out/'mhr_refit_candidate.npz.part').replace(out/'mhr_refit_candidate.npz')
 copy(c,remote+'/refit_output/refit_report.json',out/'refit_report.json')
 assert json.loads((out/'refit_report.json').read_text())==manifest['refit']
record_fit(out)
PY
```

旧记录始终保留；新尝试的 anchor/当前清单不匹配时停止。若 manifest 缺少预期字段，不绕过校验，保留原件按版本契约处理。远端无自动取消/关机/清理流程。

### 9.3 只审计原生参数（可选云计算）

原生 audit 重放已有参数，不再执行 SAM 或重新拟合。既有 `output/multiagent_gpu/native_audit/request.json` 绑定九个 remote job 与视频/档案 hash。新审计使用新 UUID、新私有目录，把 `audit_native_batch.py` 与新请求上传；远端命令格式：

```text
/root/tennis-sam3d/venv/bin/python <audit-dir>/audit_native_batch.py
  --root /root/tennis-viewer --request <audit-dir>/request.json
  --output <audit-dir>/reports --batch-size 16
```

脚本使用非阻塞共同锁；GPU 正忙即退出，不能另起绕锁的进程。下载报告和日志后记录文件 hash，检查每片 verified_replay、源/拓扑/帧数身份与 real/mirror gate，再登记：

```sh
python3 -B viewer/video_import/mhr_fit_progress.py \
  --audit-report output/multiagent_gpu/native_audit/reports/native_audit_report.json
```

历史 audit `c3d636e…` 9/9 回放通过，42.63 秒，最大网格欧氏差 1.0196e-6 m；它单独记 audit，不计通过联合拟合数量。实际请求/脚本/云命令和下载记录见 [MULTIAGENT_GPU_ITERATION.md](MULTIAGENT_GPU_ITERATION.md)。

## 10. 发布、验证、保存与迁移

### 10.1 本地部署与最小验证

只修改文档/台账脚本不必重建 UI；改变 Viewer 资产后才执行本地部署：

```sh
python3 -B deploy/3dpose/prepare_local.py
VIEWER_BASE_IMAGE=nginx:stable-alpine docker compose -p tennis-3dpose-local \
  -f output/3dpose_local/compose.yaml --env-file output/3dpose_local/.env up -d --build
```

视频库只读挂载，API 由 Mac 18768 提供；结果本地 18769 打开。复现使用旧镜像 digest 时单独保留其 inspect 信息，`nginx:stable-alpine` 是浮动标签。公网 bakewell.cloud 不属于本手册部署动作。

```sh
python3 -B -m unittest discover -s viewer/video_import -p 'test_*.py' -v
node viewer/video_import/test_racket_layer.cjs
```

检查真人/背面/左右，所有模式、播放/暂停/拖动、快速挥拍、最后一帧，录下真实 displayed frame。帧数、mask 非空、回放一致、UV 覆盖和真实拟合精度分别验收。测试是合同/回归验证，不代替真实视频 GPU 和人工姿态真值。

主 Viewer 球拍应保持既有 SHA `20220f335990618508316a814a70ae1c82b072871205a53d4ebc7b6f5e684f12`，除非有明确通过的发布替代；本手册没有替换它。

### 10.2 每轮必须保存

| 层 | 文件/记录 |
| --- | --- |
| 输入 | 原片、归一化片、SHA、实际解码帧数/fps/尺寸、输入清单 |
| 环境 | Python/依赖清单、SAM 源码/补丁、模型/配置 SHA、CUDA/driver、实际 Worker 源码 |
| 推理 | remote job、request、status/manifest、日志、完整 NPZ，不能只存导出 mesh |
| 几何 | 相机/镜面、标定角点、手/拍柄/拍框标注、尺寸 measured/assumed、冻结留出列表 |
| 纹理 | 每片 observations NPZ+JSON、UV、PNG、sources、报告、算法源码快照 |
| 拟合 | 输入 staging/repacked、候选、refit report、contact comparison、anchor/completion、总台账全部 events |
| 评审 | 选帧表、多角度截图与 hash、验收决定、未通过原因 |

九段批数据、审核数据和主片 iterations 位于忽略目录；外部备份空间至少覆盖这些完整文件和模型。不要只备份报告 JSON，因为它们引用的视频/NPZ/缓存不在 Git。

可将完整的源码历史另存 bundle：

```sh
git bundle create "$RUN/source.bundle" HEAD
```

大数据备份示例（**只作为手动备份命令，不在本轮自动执行**；输出放到独立外部路径，避免把压缩包再纳入自身）：

```sh
tar -czf /absolute/external-backup/tennis-evidence-2026-10-01.tgz \
  output/multivideo_texture output/multiagent_gpu output/video_sufficiency \
  output/mhr_fit_progress output/reproduction/2026-10-01 \
  output/video_library/85ade7a072984579831f5cb76e8e5fd3
```

SSH 凭证和 `host.local.json` 用私有配置备份；模型权重和已安装 SAM 源码另行保留。报告 hash 不能恢复文件内容。权重取得方式仍按模型访问权限处理。

### 10.3 迁移后的路径与索引

当前 batch manifest/source_index 包含原机器的绝对 `folder`。迁移时保留原件，**在新批次副本**中把每个 folder 指向新 `clips/<stem>`，保留 video/source/archive hash；记录原/新 manifest hash 与路径映射。解码视频、NPZ、observations、UV 均不改内容，hash 应一致。缓存 code/video/archive/mirror 身份不一致时停止使用，而不是删除身份字段。

新系统无法用 Mac MPS 时，独立后处理脚本可选 cpu 或已配置 cuda；`batch_multivideo.py` 当前镜面后处理固定 mps，不能直接作为跨平台保证。新环境先跑一段并确认所有合同/源身份，再扩展批次。

## 11. 常见故障与继续条件

| 现象 | 检查/处理 | 继续条件 |
| --- | --- | --- |
| SSH 断开、界面显示失败 | 查原 job；分别用生成 collect 或拟合下载恢复 | 源/结果 hash 正确，原任务状态明确 |
| 批处理返回但部分片失败 | 看 batch_manifest 全部状态与 mirror.log | 全 ready，所需镜面/观测通过 |
| 相机 root 不一致 | 新原生档案重新包装、镜面/球拍重估；旧实验用保存 repacked | 2 mm / focal 门槛通过 |
| 背面灰色或缺纹理 | mirror_valid、人物角色、SAM2、深度、mask 与投影；不要把缺 3D 当有观察 | 新有效观察，或继续明确保留未知 |
| 拍面平滑但方向冲突 | 检查原分辨率柄底/拍喉/拍框 A/B 和 MCP/PIP | 固定留出投影、方向、接触均通过 |
| gap 改善、留出退化 | 候选 rejected，保留旧发布版本 | 完成新的独立证据或损失改进再验收 |
| PNG 2048 但脸/手仍模糊 | 查看源人物/手部像素尺寸与动态模糊 | 补清晰近景，不以放大当真实细节 |
| 当前 hash 不等于历史报告 | 比较原视频、代码、缓存、模型、解码器/环境 | 确认版本后作为新运行记录，不覆盖旧证据 |

## 12. 本轮记录与相关资料

本轮只读保存环境/代码/模型身份，复算现有固定留出颜色对照；没有重新推理九段、运行 MHR 优化、部署远端 Worker 或更新公网。环境快照与本地复算原件在 `output/reproduction/2026-10-01/`，机器摘要在 [REPRODUCTION_BASELINE.json](REPRODUCTION_BASELINE.json)。

实际本地复算 `local_comparison/comparison_report.json` 与历史颜色对照的 fixed baseline/candidate、邻面指标和 face support 完全相同：1,463,590 个 tuple，MAE **19.715998408619264 → 12.831933993544299**。本轮使用 `--compare-only`，没有重新渲染候选 atlas；不能据此宣称已再次复现全部 GPU/PNG 产物。

本轮另核对 19 个 CLI 的 `--help`、24 段 shell 语法、8 段 Python 示例语法及本地文档链接，校验记录保存在环境快照目录。部署/生成/拟合/备份命令仅记录，未作为本轮执行项。后续执行从新 round 目录开始，保存实际参数、运行环境、日志、报告与验收结论。

- [部署组件与首次云闭环](CLOUD_GPU_PLAN.md)
- [真人/镜中 SAM3D 与双 SAM2 契约](MULTIVIEW_GPU_ITERATION.md)
- [球拍关键点、方向和验收](RACKET_DIRECTION_ITERATION.md)
- [多视频数据与纹理历史](MULTIVIDEO_TEXTURE_ITERATION.md)
- [多 Agent 云审计与固定留出对照](MULTIAGENT_GPU_ITERATION.md)
- [MHR 完成度与后续实施](MHR_FIT_PROGRESS.md)
- [可复用球拍拟合 skill](../../skills/tennis-racket-fitting/SKILL.md)

## 2026-10-01：完整 Viewer 接入九视频纹理与球拍同步修复

用户截图为 `85ade7a072984579831f5cb76e8e5fd3` 的第 106 帧（零基 105，4.20 秒）。此前最新纹理仅发布到独立 `5242a81d6773428090c4ed5274f3015b` 评审页，完整 Viewer 没有 atlas 绑定，因此打开完整 Viewer 不会自动看到新纹理。

现在完整页默认使用已绑定的多视频合成候选，支持切回视频投影。UV seam 顶点通过 `appearance_map.bin` 映射当前显示姿态（raw/smooth/refined），深度与诊断仍保留原始网格。绑定要求源视频出现在纹理输入清单中、三角面逐项一致、布局及 PNG 哈希一致；浏览器再次校验资源哈希和拓扑。灰色区域仍为无观测区。这里的候选展示不是全身几何或真实新视角精度验收。

复现发布：

```sh
python3 viewer/video_import/publish_dataset_appearance.py \
  --result output/video_library/85ade7a072984579831f5cb76e8e5fd3/result \
  --fusion output/multiagent_gpu/texture_quality/consistency_v1 \
  --layout output/multivideo_texture/mhr_uv.npz
node viewer/video_import/test_dataset_appearance.cjs
node viewer/video_import/test_racket_layer.cjs
python3 -m unittest discover -s viewer/video_import -p test_dataset_appearance.py
python3 -B deploy/3dpose/prepare_local.py
VIEWER_BASE_IMAGE=nginx:stable-alpine docker compose -p tennis-3dpose-local \
  -f output/3dpose_local/compose.yaml --env-file output/3dpose_local/.env up -d --build
```

发布脚本只写 appearance 文件和 viewer.html，既有文件备份在 `result/appearance_previous/`；不写人体网格、球拍姿态或人工标记。不要把这一纹理绑定到清单外的人或衣着。原生 18768 服务要重启才能识别新增 JS 路由；18769 的 Docker 静态资源已包含该 JS，无需为它中断 Mac API。

截图球拍诊断：

- 第 106 帧旧球拍 `quality=constrained_estimate`，握柄与手部先验方向夹角 91.9005°，不是独立真值角误差。
- 相对 MCP/PIP 推算握点的差异 7.2912 mm；raw→refined 顶点变化最大 7.9858 mm，不能解释大的方向冲突。
- 矩阵按行写入 Three Matrix4，与 Python R.tolist 一致；模型本地柄握点 y=0.045 m、长度 0.685 m，没有发现行列转置或模型原点解释错误。
- 全段旧方向先验夹角中位 39.4645°，P95 128.7026°；第 106 帧更新候选仍为 55.7189°。既有 racket_quality_gate 标记 needs_review，更新候选的部分投影指标退步。因此本轮没有把未通过检查的候选发布成默认姿态。
- 播放有可复现的显示缺陷：旧代码按 video.currentTime 让球拍插值到下一帧，人体仍按 requestVideoFrameCallback 的已解码帧显示。新增 CPU 回归先失败，修复后通过。现在球拍和人体共享已解码帧；没有改变或声称改进静态拟合准确度。
- “双视角握拍”改名“双视角手部”，避免把小幅手部约束预览表达成握拍验收；明显方向冲突在主界面显示，角度是手部估计与拍柄方向的差异。

后续拟合应保持当前纹理/姿态独立版本：复核高置信拍柄端点、拍喉和手指对应；把手部可靠性作为权重而非硬真值；在接触、拍框重投影、有向拍柄与时间连续性上共同优化。继续用固定 heldout 与手部接触/运动指标验收，不能仅靠轮廓或掌心零距离宣布可靠。

本轮实测：浏览器完整页默认多视频合成；第 106 帧正背视角及投影切换可用；HTTP atlas 四个文件均匹配发布清单。图索引重建仍因本机缺少 graphify 模块失败，未把该步骤计为通过。

## grip_refit_v7：手指接触、拍柄方向与拍框观测联合目标

准备阶段（2026-10-01）：代码和本地预检完成后，云 SSH 曾两次被远端关闭，初始记录为 `prepared_gpu_unreachable`。随后连接恢复，已完成真实 GPU 拟合并取回候选，最终状态为 **completed_rejected**；具体结果见下节。Viewer 球拍未被替换。

实现位于 `grip_objective.py` 和 `refit_fullbody.py`：

1. 保留原生 MHR 回放一致性门槛；形状、尺度、相机根节点保持不变，优化骨骼参数和刚性球拍。
2. 掌内 MCP/PIP 握点拉近之外，对近端关节与假设圆柱的径向距离做双向损失，处理分离与穿入；另限制接触位于有限柄段 `[0, 0.16] m`。
3. 加入有向 MCP/PIP 拍柄轴损失和观测 `handle_end→tip` 的有向图像损失，180° 反向不再视为等价。关节轴与清晰训练观测冲突超过 35° 时，手部方向权重降为 0.15；保留帧不会参与权重计算。
4. 拍框仍通过已有可见 tip、rim 和 head_center 等关键点的稳健重投影约束；本轮没有新增独立拍框标注，也没有把不可见端点补成真值。真人、镜中观测分别投影；误差按 1280 宽统一归一化。
5. 报告分别记录接触与方向前后指标、真人/镜中保留帧误差和运动步进。验收要求既有保留帧改善至少 10%、身体位移 P95 小于 3 cm，并且接触/手部方向/旋转步进/角加速度与保留帧各项 P95 不退步。缺少镜中或有向端点样本时记录零样本，不视为该项已验证。
6. 接触仍是**近端关节—假设圆柱代理**，不是手部皮肤网格表面接触；标准尺寸不是实测值，真实握柄棱位仍未校准。

固定输入：复用 v6 原生相机重打包的私有 JSON，保存到 `iterations/grip_refit_v7/staged/result/`，不读取用户后来可能编辑的标记。源原生归档 SHA 为 `1121e7d61b5d625cf396694d8fee26904ab1dfea273eef16315f522d0ce36451`。训练零基帧 `[41,65,86,100,174,185,205,236]`，保留帧 `[17,126]`，均为未人工确认的自动轮廓证据。仅两帧保留集不能证明整个视频或其他场景的准确率。

云连接恢复后，先确认没有其他生成作业，再部署并执行：

```sh
python3 deploy/3dpose/cloud_gpu/deploy.py --config deploy/3dpose/cloud_gpu/host.local.json
python3 viewer/video_import/run_fullbody_refit.py \
  --dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v7/staged \
  --native-output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native \
  --output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v7/gpu-fit \
  --config deploy/3dpose/cloud_gpu/host.local.json --allow-assumed --allow-automatic
```

`deploy.py` 已加入 `grip_objective.py`，GPU worker 的代码清单和返回报告记录该文件哈希。运行器会自动写远端 job、完成度和不可变进度事件；返回候选仍保存在私有目录。完成后比较固定保留集、接触和方向报告，再制作包含人体新网格与球拍的新旧对比，进行正背左右斜视和握拍特写检查；通过后才发布。

本地验证命令：

```sh
python3 -m unittest discover -s viewer/video_import -p test_grip_objective.py -v
python3 -m unittest discover -s viewer/video_import -p test_refit_inputs.py -v
```

5 项新增验证涵盖脱手恢复梯度、有限柄段、有向轴、保留集隔离、人体/球拍梯度与 CPU 合成小例子的联合优化完整路径；另有 6 项既有输入/原生回放验证通过。CPU 合成测试不是本视频的真实 MHR 重拟合。图索引重建因缺少 graphify 未完成。


### v7 实际云运行与验收（2026-10-01 19:33–19:35，UTC+8）

- 远端 job：`1d61ff710c8c4204a2b8e89988bb6a0e`；父原生 job：`cfecbc204433489d9cc7b20c8fa8cb1b`。
- RTX 4090 执行 120 步，worker 运行时间约 86 秒；本地已收到约 48 MiB 候选及报告、日志、远端清单和完成度记录。
- 执行代码 Git：`60bd4102eba02a7aaf563774ea9806b156d9603f`。
- `refit_fullbody.py` SHA：`031c0da984b50302044a64c9c05f8dc21c20f710d0b73795c857711b0cbe61f6`。
- `grip_objective.py` SHA：`1ca4cadff8ad4fd41ac51420c158f319b594d9000ec60268fc7186a59bebd802`。
- 候选 SHA：`b3b52cdc07d9f3c750275d7b55da7903282dc5d6364cc5587b9fa2b14ac66e14`，与本地文件、报告和远端清单一致。
- 原生回放最大误差 `7.15256e-7 m`；人体位移 P95 `15.4015 mm`。这两项通过不代表握拍准确。

| 指标 | 初值 | v7 结果 | 判定 |
| --- | ---: | ---: | --- |
| 掌内握点距离中位数 / mm | 6.4495 | 0.5988 | 改善 |
| 手指—柄圆柱径向差中位数 / mm | 7.5637 | 2.2488 | 改善，接触代理 |
| 手部先验—拍柄夹角中位数 / ° | 32.1761 | 8.9832 | 改善，非独立方向真值 |
| 保留帧关键点误差中位数 / 1280 宽像素 | 15.2390 | 35.4743 | 退步 |
| 保留帧关键点误差 P95 / 1280 宽像素 | 29.2191 | 52.9491 | 退步 |
| 保留帧有向拍柄误差 / ° | 8.6005 | 14.0629 | 退步，仅 1 组端点 |
| 旋转步进 P95 / °每帧 | 22.2864 | 28.3452 | 退步 |
| 角加速度 P95 / °每帧² | 4.7288 | 16.6319 | 退步 |

初值是 v6 固定原生相机重打包中的方向候选，并非当前 Viewer 的默认 v4 姿态；此表与 v6 重拟合比较使用同一初值、输入和保留帧。保留集只有 2 帧、7 个关键点、1 组有向端点，没有镜中保留观测，不能据此证明泛化精度。

结论：**计算完成，验收拒绝**。手部贴合改善不能抵消视频球拍位置与抖动退步。`completion.json` 和全局台账已自动记录 completed / rejected；`iteration_manifest.json` 已更新，保留原输入与代码哈希。当前 Viewer 默认球拍 SHA 仍为 `20220f335990618508316a814a70ae1c82b072871205a53d4ebc7b6f5e684f12`；合成纹理不受本轮影响。候选因数值门槛失败没有进入发布用多角度视觉验收，不能记为视觉验收通过。

下一步实施依据：当前只有 8 个训练帧提供图像约束，全段 249 帧只有 2 帧手部方向降权，而关节接触/方向在全段施加；这提示观测稀疏与手部先验权重不平衡。下一轮应先增加并复核训练段的可见拍柄/拍框观测、隔离镜中角色，并用置信度处理无观测区间；同时对实际球拍旋转施加时间约束（当前正则主要针对参数修正量）。不应通过提高平滑或降低验收门槛宣称成功，也不应反复用这两个已查看的保留帧调参后称其为独立测试；最终验收需补充未参与调参的观察集。

## grip_refit_v8：用户重新标记后的观测与置信度

输入快照位于 `iterations/grip_refit_v8/staged/result/`。本次读取 10 帧人工标记：8 帧完整真人拍柄/拍框、4 帧包含镜中观测；球拍尺寸仍为标准假设，物理拍面正反侧未确认。人工标记 SHA、尺寸、关键点、原 Viewer 球拍/网格/纹理 SHA 保存到 `iteration_manifest.json`，运行时不再读取随后可能变动的现场标记。

本次冻结的验证集为零基帧 `[17,185]`（界面第 18、186 帧），整帧真人和镜中观测均不参加训练。训练共 221 帧；总观测集合包含 10 帧人工标记及 213 帧自动观测。验证点与 v7 已不同，不能用 v7 的 15.24 像素直接当成本轮初值。比较应对同一份新人工验证集分别计算初值和候选。

实现变更：

- `refit_observations.py` 校验自动点的视频、时间轴、尺寸及关键点文件哈希。置信度低于 0.2 的自动视图不采用；权重上限 0.3，并随定位不确定性降低。
- 人工帧整帧覆盖自动结果；人工未标出的点/视角保留缺失，不能把自动补点继承成“人工确认”。人工点权重为 1。
- 未经证实的手部方向基础权重为 0.1；与训练观测明显冲突时为 0.03、相符时为 0.4。接触代理亦适度降权，保留独立人体先验。
- 未确认物理侧的 rim_side/rim_opposite 按无序点对拟合和评估，不能据此声称辨认真实拍面 A/B。
- 增加实际相邻球拍旋转的变化约束，避免只正则化参数修正量。
- 密集观测首跑发现 batch 内图像残差累加而身体先验取平均；归一化复跑修正为每个观测帧贡献一次。CPU 测试确认 batch size 2 与 16 的球拍输出一致。该修复依据目标函数尺度，不改变本轮验证集和验收阈值。

首跑代码 `d939370`，job `74e4a9129cf04f68b1f371a4e80674ca`，输出 `gpu-fit/`。归一化修复代码 `d3e2158`，复跑输出 `gpu-fit-normalized/`；两次代码快照分别存放于 `code_snapshot/`、`code_normalized_snapshot/`，不覆盖历史代码。

复跑命令：

```sh
python3 deploy/3dpose/cloud_gpu/deploy.py --config deploy/3dpose/cloud_gpu/host.local.json
python3 viewer/video_import/run_fullbody_refit.py \
  --dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v8/staged \
  --native-output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native \
  --output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v8/gpu-fit-normalized \
  --config deploy/3dpose/cloud_gpu/host.local.json --allow-assumed --allow-automatic
```

历史复现需使用对应代码快照，不能直接以未来代码覆盖历史输出目录。保留原生相机重打包的镜面平面，其距离与现场 Viewer 平面相差约 0.14 mm、法线差很小；不把旧 Viewer 相机与新原生 root 混用。

本地 29 项相关测试通过：5 项接触/方向/梯度/批次一致性测试、2 项人工优先和哈希/留出隔离测试、6 项输入/原生回放测试、16 项运行记录测试。Graphify 仍因本机缺少模块未完成。

### v8 实际结果：两次均执行完成，归一化版仍因运动指标拒绝

首次 job `74e4a9129cf04f68b1f371a4e80674ca`：120 步完成，候选 SHA `2c2a16d7f8242710c9e9443a3cd162ca7d6e77e5229d80c634c480f08901ab71`。真人保留关键点中位误差 29.2630 → 10.3394 canonical px，身体位移 P95 30.6841 mm，角加速度 P95 22.2490°/帧²，验收拒绝；该结果保留为归一化缺陷修复前的记录。

归一化 job `c30a316afce148e69cfac898cf2ea963`：2026-10-01 20:27:56–20:30:42（UTC+8），RTX 4090 完成 120 步。候选已取回，SHA `be823e8e049414970d2235fc39f0c23f18bc3a97058c949eb21ba1e7de53a905`。代码和输入均已与本地冻结快照核对；两次运行的 completion 与总台账分别保存 completed / rejected。

| 指标 | 同一初值 | 归一化 v8 | 结论 |
| --- | ---: | ---: | --- |
| 真人保留关键点误差中位数 / canonical px | 29.2630 | 7.6270 | 降低 73.9% |
| 真人保留关键点误差 P95 / canonical px | 41.8010 | 16.9244 | 改善 |
| 真人保留柄轴夹角中位数 / ° | 17.2610 | 8.9214 | 改善，2 组端点 |
| 镜中保留关键点误差中位数 / canonical px | 20.8310 | 20.1326 | 小幅改善，5 点 |
| 镜中保留柄轴夹角 / ° | 32.7381 | 1.0416 | 改善，仅 1 组端点 |
| 掌内握点距离中位数 / mm | 6.4495 | 4.2652 | 改善，代理指标 |
| 手指径向距离差中位数 / mm | 7.5637 | 3.5805 | 改善，代理指标 |
| 手部估计与柄轴夹角中位数 / ° | 32.1761 | 27.1266 | 改善但仍存在冲突 |
| 旋转步进 P95 / °每帧 | 22.2864 | 26.7738 | 退步 |
| 角加速度 P95 / °每帧² | 4.7288 | 9.2368 | 退步 |

原生回放最大误差 `7.15256e-7 m`；身体位移 P95 `17.5452 mm`，小于 30 mm 门槛。归一化版的保留观测分项 gate 通过，整体数值 gate 因运动指标退步仍失败。数据表明人工校准和置信度处理改善了视频中的位置/方向；它没有证明真实握柄棱位、手部表面接触或整个视频的准确率已达标。

默认 Viewer 的人体网格、球拍、最新合成纹理以及现场人工标记均与本轮执行前哈希一致。候选因运动门槛失败未进入发布用视觉验收，也未替换主 Viewer。未修改门槛让它通过。

下一轮优先定位旋转加速度异常的具体区间与观测切换：区分真实挥拍转向、拍框对称解切换和自动观测跳变，再以观测置信度门控实际旋转的连续性；不应仅加大整段平滑。当前两帧保留集已用于本轮结果分析，后续最终验收还需新增未参与调参的标注帧。

## v9 准备：抖动与观测切换审计（尚未执行 GPU 拟合）

分析对象为 v8 归一化候选，使用 SciPy SO(3) 独立重算旋转步进和加速度。可复现命令：

```sh
python3 viewer/video_import/audit_racket_jitter.py \
  --candidate output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v8/gpu-fit-normalized/mhr_refit_candidate.npz \
  --observations output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v8/observations_snapshot.json \
  --output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v8/jitter_audit.json
```

结果：相邻拍面法线反向次数为 0，单帧旋转最大 36.9000°，不支持“瞬间 180° 拍面翻转”是当前主要原因；这不排除更缓慢的对称解漂移，也不证明真实物理拍面身份已识别。

最大的 15 个加速度位置有 13 个位于观测边界：包括观测缺失/恢复、可见关键点集合变化、人工/自动来源交接。全段 247 个可评估中心中本来就有 176 个边界，因此 13/15 只是定位线索，不能单独证明切换导致抖动。最高的三个位置为界面第 183、106、116 帧，加速度约 14.4、14.2、11.9°/帧²。第 3 帧附近自动拍头的相邻中点残差约 40 canonical px，但前后也在变化，未将其自动判作错误点。

新增保守门控 `gate_isolated_points`：只有中间帧及前后相邻帧都属于自动训练观测，前后端点距离小于 12 canonical px、中间与两侧均相差超过 12 px，才将该点权重乘以 0.1。不会插值/补点，不改人工标记，不读取保留帧或跨越人工帧来判断异常。正常连续快速移动不满足该门槛。

本数据仅 1 点满足门槛：零基第 94 帧（界面 95 帧）的镜中 throat，中点残差 16.7064 canonical px。它并不覆盖大多数高抖动区间，不能声称该门控已解决整体抖动。下一步应检查高抖动区间的原画面、角色关联和可见部位交接，避免把真实挥拍或正确人工标记抹平。

8 项相关测试通过（5 项目标/批次一致性，3 项观测测试，包含新门控对人工/保留帧隔离和快速运动保护）。v9 输入、代码和门控记录保存到 `iterations/grip_refit_v9/`；沿用 v8 冻结输入和验证集。云 SSH 再次返回 Connection closed，部署未成功，也未调度 v9 云 job。**尚无门控后 GPU 结果，未更新 Viewer，未新增已完成 MHR 拟合计数。** Graphify 仍因缺少模块未完成。

恢复连接后：先部署当前版本，再用 `run_fullbody_refit.py --dataset .../iterations/grip_refit_v9/staged --native-output .../iterations/grip_refit_v6/native --output .../iterations/grip_refit_v9/gpu-fit --config deploy/3dpose/cloud_gpu/host.local.json --allow-assumed --allow-automatic` 执行。保留 v8 对照、固定观测和原阈值，报告单点门控的真实效果，不预设改善。

### v9 云恢复后的实际结果（2026-10-01 21:29–21:31，UTC+8）

先前的连接阻塞已解除，本次成功执行 job `62da04039fa145cb8f4b385478b68f25`，RTX 4090 运行 120 步并返回约 48 MiB 候选。部署代码为 `3ae022b200c1d9e724a0cda6b596a475421dbe5a`；候选 SHA `f3a4c2eea2e3c0f8ca850bc2fe9e823420cd63671d8e15d4af19107bc7bf492b` 已与远端清单、报告及本地文件核对。执行完成，验收拒绝；自动完成度台账已更新。

使用 `compare_refit_runs.py` 对 v8 归一化版与 v9 比较。该脚本要求源视频/原生档案、训练和保留帧相同，且冻结的人体元数据、镜面、人工标记、自动点、尺寸和初始球拍 JSON 哈希完全相同，候选文件通过哈希核验后才输出差值。

```sh
python3 viewer/video_import/compare_refit_runs.py \
  --before output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v8/gpu-fit-normalized \
  --after output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/gpu-fit \
  --output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/comparison.json
```

| 指标 | v8 归一化 | v9 单点降权 |
| --- | ---: | ---: |
| 角加速度 P95 / °每帧² | 9.236806 | 9.236820 |
| 旋转步进 P95 / °每帧 | 26.773760 | 26.750616 |
| 真人保留点误差中位数 / canonical px | 7.627023 | 7.626994 |
| 掌内握点距离中位数 / mm | 4.265179 | 4.264485 |
| 身体位移 P95 / mm | 17.545190 | 17.528873 |

变化很小，没有整体抖动改善证据；没有测量 GPU 重复运行噪声，不能把微小差值当作有效提升。只有一个镜中拍喉点被降权，实验说明处理该点不足以解释或解决主要抖动，并不排除其他类型的自动观测错误。

本轮没有替换 Viewer 或更改发布门槛。下一步应集中检查观测缺失/恢复与人工/自动约束交接处，按原视频分辨真实运动与观测不连续，再设计区间连续性约束。先前审计的“13/15 靠近边界”仍是相关线索，不应直接当作因果结论。新增比较脚本实际运行成功；Graphify 仍因本机缺少模块未完成。

### 按用户明确要求：当前 Viewer 切换到 v9 候选预览

用户明确要求直接替换 Viewer 查看效果。已将 v9 的人体与球拍作为配套候选接入原地址，仍保留 `accepted=false` 与页面“候选预览 · 抖动未通过验收”标记。此次是用户指定预览，不改变历史拟合验收结论。

发布命令：

```sh
python3 viewer/video_import/publish_refit_preview.py \
  --result output/video_library/85ade7a072984579831f5cb76e8e5fd3/result \
  --run output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/gpu-fit
```

备份位置：`output/video_library/85ade7a072984579831f5cb76e8e5fd3/preview_backups/before_joint_20261001_214046/`。备份包含原 mesh_refined.bin、mesh_meta.json、racket_poses.json、viewer.html 及哈希清单。人工球拍/地面标记、原始投影网格、源视频、合成纹理均保留。

显示网格用 `candidate_local + candidate_root - existing_source_root` 重定位，保持既有视频投影坐标，同时让新人体与新球拍使用同一世界坐标。实际验证最大世界坐标差为 `4.76837e-7 m`。当前页面固定配套 v9 姿态，移除旧球拍候选切换/局部重拟合入口，避免与新人体混配；原标注文件仍在，恢复原页面即可继续原编辑流程。

HTTP 已核对新 meta/球拍/HTML 与本地文件一致；浏览器实际显示“联合拟合 v9”和默认多视频合成纹理。当前部署是 Docker 对结果目录的实时挂载，无需重建镜像。`gpu-fit/publication_event.json` 与迭代清单记录用户指定预览；历史 refit_report/completion 的计算验收记录不改写成通过。

恢复此前版本（先关闭播放，再执行；不要覆盖新增的人工标记）：

```python
from pathlib import Path
import shutil
root = Path('output/video_library/85ade7a072984579831f5cb76e8e5fd3')
backup = root / 'preview_backups/before_joint_20261001_214046'
for name in ['mesh_refined.bin', 'mesh_meta.json', 'racket_poses.json', 'viewer.html']:
    pending = root / 'result' / (name + '.restore')
    shutil.copy2(backup / name, pending)
    pending.replace(root / 'result' / name)
(root / 'result/joint_preview_manifest.json').unlink(missing_ok=True)
```

## v10：连续手方向先验对照（2026-10-01 22:53–22:55，UTC+8）

新会话接续检查 v9 Viewer：从首帧播放后界面到达 249/249 并停止；这证明时间轴可完整推进，不等于逐帧录像审计。保存第 106、116、183 帧的正面、背面、左右、斜视和握拍特写到 `iterations/grip_refit_v10/viewer_review/`，截图 SHA 与局限写入 `review.json`。183 帧拍柄靠近手指，但手形/拍柄方向仍不一致；近距离画面不能证明皮肤接触。页面保持 v9 未验收候选和九视频合成纹理。

热点 zero105/115/182 附近存在真人拍柄点缺失、镜中点增减。另发现旧手方向先验在 35° 处由 .4 跳到 .03。v10 **只**将该角度门控换成 25–45° 间的 smoothstep；清楚一致/冲突的 .4/.03 端点与缺证据的 .1 默认值保留，没有时间平滑或补点。只改变 10 帧先验，保留帧不变，人工标记和自动点权重保持原样。参数区间是实验设定，尚未证明能改善拟合。

云 GPU 无活动计算进程且 `gpu.lock` 可非阻塞取得后部署，执行 job `474e4c2a51874b8aaafdbe9f5163435e`，120 步，候选 SHA `7a1c65a7fd3813f019c2dda6fdfbefa8670d25ff26f952fb504ebe7231eff0bc`。本地/远端/报告哈希已核验；输入继续冻结为 v9 staged，代码保存在 v10 `code_snapshot/`。部署 Git revision 指向此前提交，实际实验代码身份以清单 SHA 为准。

| 指标 | v9 | v10 |
| --- | ---: | ---: |
| 角加速度 P95 / °每帧² | 9.236820 | 9.232450 |
| 旋转步进 P95 / °每帧 | 26.750616 | 26.729008 |
| 真人保留点误差中位数 / canonical px | 7.626994 | 7.627146 |
| 掌内握点代理距离中位数 / mm | 4.264485 | 4.239344 |
| 手/拍柄方向夹角中位数 / ° | 27.126366 | 27.314934 |
| 身体位移 P95 / mm | 17.528873 | 17.528062 |

结果仍为 **completed_rejected**，未发布 v10。抖动变化极小，手方向中位数略恶化，没有整体改善证据；GPU 重复噪声仍未量化。独立 SciPy SO(3) 审计得到最大单步 36.95698°、加速度 P95 9.23248°/帧²、相邻法向反转 0，top15 中 13 个靠近证据边界（全部中心的 176/247 本就属于边界）。这些关联不能证明因果。

v9 相对冻结初始球拍的拍面法向最大变化约 67.89°，没有变成相反半球；这没有验证初始物理正反面，也没有排除初始轨迹里的缓慢对称漂移。接续优先量化拍柄/拍框证据消失与恢复的权重变化：zero180–182 只有镜中拍头/拍尖，zero183 镜中拍喉恢复，zero184 真人拍柄恢复。不应继续把 35° 门控当成主因或仅增强全局平滑。

复现取回后的固定输入比较与独立审计：

```sh
python3 viewer/video_import/compare_refit_runs.py \
  --before output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/gpu-fit \
  --after output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v10/gpu-fit \
  --output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v10/comparison.json
python3 viewer/video_import/audit_racket_jitter.py \
  --candidate output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v10/gpu-fit/mhr_refit_candidate.npz \
  --observations output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v10/observations_snapshot.json \
  --output output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v10/jitter_audit.json
```

本轮 15 项相关测试通过（目标函数 6、观测 3、输入 6），包含连续门控端点/邻域、保留帧隔离、CPU 联合拟合和批次一致性。Graphify 按要求再次尝试，仍因缺少模块失败，未安装。完成度台账已记录计算完成、验收拒绝，不能增加已验收 MHR 联合拟合数量。

## 2026-10-02 顺序验证接续

新增纹理已经完成九段留出纹素投影审计、三段同几何视觉抽查，以及 v9/v10 球拍证据边界审计。具体命令、表格和独立预览见 [离线外观迭代的顺序验证记录](OFFLINE_APPEARANCE_ITERATION.md#顺序验证2026-10-02)。产物为 `output/offline_iterations/20261002_sequential_validation/`，新脚本/测试快照与报告哈希在 `validation_manifest.json`。

当前结论是验证执行完成、验收未完成：新增纹素具有独立留出支持，但颜色残差/衣物边缘/手部外观仍需处理；球拍热点有自动点缺失/恢复、置信度变化和镜中重投影冲突。没有改写已拒绝的拟合记录，没有替换主 Viewer。云 GPU 已通过空闲与锁检查，本轮尚未调度新工作。下一轮应先复核自动拍框/拍柄观测与镜面角色/几何一致性，再用固定输入的有界对照实验验证影响；不能仅靠全局平滑，也不能把外观覆盖当作几何/握拍验收。

恢复后应同时记录 publication_event 的撤回状态，刷新页面查看；不要删除历史候选或备份。


## 本地多 Agent 外观迭代（2026-10-01，跨日完成）

见 [OFFLINE_APPEARANCE_ITERATION.md](OFFLINE_APPEARANCE_ITERATION.md)：两组外观输入独立校验、可选纹素回退、缓存来源绑定与两段 CPU 对照。输出位于 `output/offline_iterations/20261001_texture_v2/`。云 GPU 未开启，未替换 Viewer；缺省组需要补齐原生参数及相机元数据。


### 九段 Viewer 集合

2026-10-02 已将剩余八段原生结果导入独立可遍历集合，支持逐帧/慢放/原生当前帧投影与共享UV纹理对照。入口、覆盖率分母和复现命令见 [OFFLINE_APPEARANCE_ITERATION.md](OFFLINE_APPEARANCE_ITERATION.md) 的九段 Viewer 章节。未替换原v9主结果，未发布为验收结果，未新增GPU任务。


## v11：镜中缺证据边界的配对消融（2026-10-02）

按顺序先复核 v9/v10 热点，再准备新标注帧，最后执行限定范围的诊断GPU对照。代码新增 `audit_racket_roles.py` 与 `prepare_racket_boundary_experiment.py`；未修改求解器、损失权重或验收门槛。

### 角色/镜面审计

使用原生真人/镜中MHR右腕作关联诊断，以固定镜面虚相机射线检查成对关键点。独立镜中MHR使用其自己的 focal，并已按原生约定恢复x；射线反射只应用一次。未确认的rim两侧不用于成对几何指标。

- 自动点：213帧、170个成对部位，极线误差中位数8.2743、P95 31.9971 canonical px。
- 既有人工标记：10帧、8个成对部位，中位数3.5501、P95 32.3776 canonical px。
- 界面第113帧自动tip射线间距0.21066m；界面第185帧自动tip间距0.18163m。
- 第186帧既有人工handle/tip间距0.16473/0.15483m。它属于既有保留帧，审计只作诊断，未修改或用于门控。
- “点更靠近另一角色的估计腕点”也出现在既有人工帧，说明腕/镜面估计本身可能有误，不能将该距离启发式直接当作角色误标分类器。未发现完全相同的真人/镜中关键点集；选中轮廓的来源ID未保留，因此不能据此证明无重复轮廓关联。

报告：`output/offline_iterations/20261002_joint_evidence_review/roles_native_focal/role_geometry_report.json`。之前的 `roles/` 是使用真人focal的初稿，已被这个使用独立mirror_focal的版本取代；两版保留以追踪差异。极线/射线指标只能说明观测与当前相机/镜面不一致，不能区分平面、相机、时间匹配和点标记的误差来源。

### 新验证帧隔离

预留零基 `[30,80,130,155,210,230]`，即界面31/81/131/156/211/231帧。A/B两组在生成新拟合前都删除这些帧的自动点与stereo_shaft，确认不在任何训练观测中。原始PNG、12个真人/镜中裁剪、哈希与空标注模板保存于 `new_annotation_packet/`。此前v8–v10使用过这些帧的自动点，因此不能称完全未见数据。新标签仍 pending，不能报告新的独立验收通过。

审核页面：

http://127.0.0.1:18769/datasets/defdd710b75a5da78951c6fd56cbbf26/result/viewer.html

该页面不显示旧自动点或拟合投影，支持原像素放大、真人/镜中关键点、逐帧检查与导出已人工检查的标注。导出不会改写主Viewer或自动触发拟合，遮挡部位保持空缺、物理A/B未确认。Browser验证帧切换、空标注导出限制，控制台无错误。

### GPU配对实验

云锁可取得、无活动计算、远端六个求解依赖SHA与本地一致后，沿用现有代码执行120步：

| 组 | job | 训练观测帧数 | 状态 |
|---|---|---:|---|
| control（两组共同隔离6个预留帧） | `7d7c903d17ba462fa8ec5192ac003e87` | 215 | completed/rejected |
| mirror_ablation（另移除zero180–184镜中自动点） | `8f73796af82a416facca9b69b0da3895` | 211 | completed/rejected |

两组真人点、人工标记、旧保留帧17/185、原生档案、相机/镜面、尺寸和初始球拍完全一致。消融仅移除5帧未确认镜中自动点；其中4帧失去所有训练点，所以training membership有4帧差异。通用 `compare_refit_runs.py` 正确拒绝此情况，未放宽其“完全同观测”检查；专用 `compare_ablation.py` 另验证所有差异只在预先声明范围，并记录观测哈希/训练帧差异。

| 指标 | control | mirror_ablation |
|---|---:|---:|
| 第183帧角加速度 / °每帧² | 14.3826 | 3.8170 |
| 第185帧角加速度 / °每帧² | 3.6862 | 7.4482 |
| 全段角加速度P95 / °每帧² | 9.1054 | 8.7182 |
| 真人旧保留点中位数 / canonical px | 7.6263 | 8.4253 |
| 真人旧保留点P95 / canonical px | 16.9230 | 25.6576 |
| 镜中旧保留点P95 / canonical px | 21.5065 | 27.7769 |
| 镜中旧保留柄轴夹角 / °（1组） | 1.0421 | 9.5045 |
| 掌内握点代理中位数 / mm | 4.1335 | 4.1603 |
| 手/柄轴夹角中位数 / ° | 27.4741 | 27.8517 |

局部抖动明显下降，但恢复交接处加速度上升、重投影和部分手部指标退步，不能采用“直接删除该段镜中约束”的方案。原视频接触表中的消融镜中投影仍明显偏离可见拍框。一次A/B没有量化GPU重复噪声，也未消除相机/镜面与标记混杂，不能声明确定因果或接受。

产物：`iterations/grip_refit_v11_boundary/` 中候选、远端清单、completion、观察快照、protocol、comparison；审计/源图/代码快照在 `output/offline_iterations/20261002_joint_evidence_review/`。两组均未发布，主v9配套预览哈希保持一致。18项相关Python测试通过，标注页面脚本语法检查通过，Graphify按Git源码范围更新。

下一步先对新预留帧填写清晰可见部位，并复核第113/185/186帧的观测对应与镜面假设；完成后单独评估新标签残差，再设计保留可信镜中观测的连续性约束。不能以本轮局部加速度改善跳过重投影和真实接触检查。


### v11：收到6帧新人工标记后的评估

已读取下载的 `reserved_validation_landmarks.json`，保存接收原件与校验副本，确认视频SHA、2560×1440、25fps、零基30/80/130/155/210/230；两组GPU报告均无训练重叠。28个真人点、9个镜中点；真人有4组handle→tip、镜中仅1组，缺失部位未补填。标签没有加入训练，不重新拟合。

| 评估指标 | 初始球拍 | v11控制 | v11镜中消融 |
|---|---:|---:|---:|
| 真人点中位数 / canonical px | 9.6853 | 8.0027 | 8.0026 |
| 真人点P95 / canonical px | 22.2073 | 17.4927 | 17.4922 |
| 真人柄轴中位数 / °（4组） | 11.3225 | 4.3892 | 4.3892 |
| 镜中点中位数 / canonical px | 17.9439 | 15.0616 | 15.0616 |
| 镜中点P95 / canonical px | 29.9607 | 28.4780 | 28.4780 |

第31帧的镜中投影明显偏離拍框；新标记的tip/throat与当前镜面两射线间距约55.9/60.4mm，第81帧tip约2.8mm、handle约28.8mm、throat约39.6mm。不能据这些诊断区分镜面/相机估计、点定位或模型尺寸问题，也不能用这些已查看标签调参后继续称其完全独立验收。两组在新6帧基本一致，消融影响局限于热点附近，不证明整体改善。

报告与同范围对照截图：`output/offline_iterations/20261002_joint_evidence_review/new_label_evaluation/`；评估脚本 `evaluate_reserved_racket.py`，3项新增隔离/零样本测试通过。整体验收仍未通过、原v9主预览保持不变。后续先用原训练证据检查镜面/观测对应，保留新6帧作回归检查；若据其调参则需另补未参与调参的最终评估数据。

新评估页面：

http://127.0.0.1:18769/datasets/f9b640731a365ceabd42bcdf0b1ad5af/result/viewer.html

### 镜面／相机与关键点对应核查：2026-10-02

冻结 v9 staged 输入和 v6 native 档案，执行 `audit_mirror_camera.py`。最新完整报告为 `output/offline_iterations/20261002_joint_evidence_review/mirror_camera_audit_v2/mirror_camera_report.json`；早先 `mirror_camera_audit/` 是缺少 COCO 反例与 FFmpeg 滤镜校验的阶段报告。输入、视频、代码哈希和 FFmpeg 重放日志均保存；未启动 GPU、未修改已保存训练观测或原 v9 Viewer。

坐标一致性通过：真人249帧、镜中245有效帧的焦距均为2937.2095px；原生相机点回投与保存2D点的最大差异分别0.000277/0.000231原图px。镜中无效zero2/6/86/170保留缺失。反射矩阵行列式为−1、二次反射误差2.22e−16，水平翻转后的x/像素已还原一次，MHR沿用解剖索引。COCO左右交换单独比较，同400个核心验证点当前映射中位15.1482px；不交换左右为45.9902px，支持当前映射。代码／坐标一致不等于几何或实际相机标定正确。

重现镜面估计的全部2633个对应点及原有核心验证中位数；核心400点P95为65.8020px，右腕28个验证点中位17.2994px、P95 85.3585px。这些均为2560×1440原图像素，换算canonical需除以2。人体纹理门槛不能作为球拍对应精度通过证明；焦距、主点、畸变和镜面距离仍未实测。

仅用既有训练zero100/174/205的5对轴向人工点、固定焦距和镜面距离，在法向±5°诊断范围内试拟合；排除旧17/185、新6预留帧及未确认rim对应。法向变化1.3528°，对称极线误差中位数如下：

| 组／原图px | 当前镜面 | 诊断法向 |
|---|---:|---:|
| 训练5对／3帧 | 2.3046 | 0.8401 |
| 旧验证3对／第186帧 | 62.5339 | 92.8464 |
| 新预留5对／第31、81帧 | 13.8872 | 19.0843 |

诊断法向没有通过验证，不替换当前镜面。极线约束不含镜面距离的绝对尺度，不能用调距离消除此类点线不一致；仅3个训练帧也不足以标定实际内参。第81帧handle_end位于手／腕附近，其是否为物理拍柄末端、两视图throat是否采用同一位置定义仍需复核。未改动用户标记，未据新标签调参；这些已查看标签作为回归数据，最终独立验收仍需另留未参与调参的数据。

发现并修正后续观测的时间映射错误：原视频为VFR，250帧，平均25.0801fps，单帧间隔约0.078–97.421ms。旧 `racket_observations.py` 用 `round(n / 25 * average_fps)` 选原视频帧，与实际归一化映射在152/249帧不同。新 `source_frame_alignment.py` 使用精确PTS与server的零起点、默认round=near的fps滤镜映射，检查归一化时间轴及EOF帧数，并保留原高分辨率取帧。对实际FFmpeg滤镜前后校验和的249帧核查全部通过；17帧抽查中时间映射正确画面的图像差异更低。

UI106/116的新映射分别为原视频zero106/116，旧为105/115；UI183两者均为183。故错帧可能解释部分热点，不能声明解释全部抖动；仍需重生成自动观测及同分割拟合对照才能量化收益。后续生成写入映射方法及归一化视频SHA，历史观测和所有v6–v11运行保留。代码已修正，本轮没有重生成观测、没有更新运行中的Docker服务镜像。

```sh
python3 -B viewer/video_import/audit_mirror_camera.py \
  --staged output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/staged/result \
  --native output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native/reconstruction.npz \
  --dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3 \
  --labels output/offline_iterations/20261002_joint_evidence_review/new_label_evaluation/labels_validated.json \
  --protocol output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v11_boundary/protocol.json \
  --output output/offline_iterations/20261002_joint_evidence_review/mirror_camera_audit_replay
```

独立核查页面（6张人工点／极线图、3张热点取帧对照及JSON）： http://127.0.0.1:18769/datasets/05f8186e7af507a887bb37e3fa4193c2/result/viewer.html 。27项相关测试与代码编译检查通过，Graphify已更新。主预览三个文件SHA与 `joint_preview_manifest.json` 一致。下一步按时间戳在隔离目录重生成观测、保留同样验证划分，随后复核明确物理对应，再考虑实测标定或联合拟合。

#### 第186帧反馈复核

用户指出第186帧未对齐，补充 `frame186_detail/` 的v9预览／v11控制／镜中消融源图对照。原图中的同色线是极线，不是拍面法线；新图绿色为人工点、黄色为拟合拍框／轴线、红色为拟合的无向法向轴，不能作为法向真值或物理A/B面证明。第186帧为zero185（7.4秒），映射原视频zero186，取帧正确。

三对人工轴向点真人侧极线偏差：handle65.9514px、throat43.9656px、tip62.5339px；镜中侧34.2313/22.5740/31.9609px，均为原图像素。当前固定相机／镜面下，改变球拍姿态不能让这三对不一致的像素同时精确对应。v9拍框在两视图仍明显偏离源图；v11控制虽镜中柄轴误差1.0421°，五个镜中点位置仍偏31.6278–43.6476原图px。需要先复核点的物理含义及镜面／相机，不能单帧旋转法向后宣称通过。

补充页面 http://127.0.0.1:18769/datasets/05f8186e7af507a887bb37e3fa4193c2/result/frame186.html ，在原核查页增加入口。报告明确 `point_error_original_px`，修正前数值相同但单位字段误用canonical的阶段报告保留为 `report_before_units_fix.json`，不作最新引用。没有优化、改标记、改相机／镜面或替换主预览；三个候选旋转正交性及帧映射核查通过、5个新增页面／数据／PNG路由均200。

#### 球拍参数与人工点差异修正（2026-10-02）

后续GPU优化没有训练第186帧；早期CPU方向拟合则曾读取人工点。发现旧CPU把人工点合并进自动点，而自动head_center以6倍残差继续参与，自动轮廓／stereo_shaft也未排除；未确认的拍框两侧固定排序，30°初始化限幅与SO(3)展示平滑还能移动人工姿态。第186帧本身取帧正确，自动拍尖距人工仅约6原图px，不能把最终姿态偏差全部归因于识别或错帧。

代码调整：`racket_manual_evidence.py` 让已审核帧完全替代自动点，人工未标出的视图／遮挡点不补自动证据；两侧物理对应未确认时各视图独立作无序匹配。`fit_dataset_grip.py` 排除人工帧自动轮廓和立体柄轴先验，使用人工柄轴判断手方向冲突，跳过人工初始化限幅。`racket_stability.py` 新增fixed_frames，精确保留这些帧求解旋转。新增显式训练编辑CLI `--manual-correction-frames 185`，以现有握点拟合真人点并在序列优化和后续平滑中固定该姿态。默认未启用强制编辑，GPU验证协议不变。

单帧工具 `correct_reviewed_racket_frame.py` 使用原图内参、正确主点、多初值和未确认两侧排列，分别诊断真人、两视图等权、固定握点三种模式。原v9固定人体下，真人五点中位22.112→3.069原图px，柄末端仍21.909px，镜中中位32.537px；固定握点gap为0，只代表守住原解剖先验，不等于实测皮肤接触。仅贴真人点中位1.305px，却需要平移893.9mm、偏离原握点853.9mm。相机／镜面／拍形／人体深度均冻结，这个冲突不能确定归罪于某个参数。

实际CPU运行位于 `output/offline_iterations/20261002_joint_evidence_review/manual_parameter_optimization/` 的before、after、after_edit185。所有运行使用同一份v9 staged JSON与该视频attempt0004原生人体，未载入multiview_constraints，未重生成历史自动时间映射；不是与v9 GPU解的直接对照。protocol记录输入SHA、实际fit脚本SHA，code_before保存旧脚本、code_final保存最终代码；旧脚本调用新版stable_rotations的默认无固定帧模式，其目标和变量与原版一致。

| CPU对照 | 真人全部人工点中位／原图px | 真人P95／原图px | 角加速度P95／°每帧² |
|---|---:|---:|---:|
| before | 28.9703 | 93.6538 | 5.0386 |
| after：人工优先 | 18.2630 | 107.5752 | 6.4412 |
| after_edit185：额外固定第186帧 | 见manual_comparison.json | 见manual_comparison.json | 6.4890 |

CPU第186帧真人点中位57.3989→2.8666原图px，柄轴17.0383→0.0095°，镜中仍32.3088px；最终旋转与显式编辑结果相同、平滑改变量0。三次整段均needs_review/rejected，门控恢复原发布姿态，诊断读取 `racket_poses_directional.json`。这轮旧10个人工帧是训练点，zero17/185历史验证角色对这些编辑候选撤销；新6帧未载入，不能把上述数字当作独立验证通过。

复现时只对新的隔离result目录运行：

```sh
python3 viewer/video_import/correct_reviewed_racket_frame.py \
  --dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3 \
  --labels output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/staged/result/racket_landmarks.json \
  --frame 185 --output output/offline_iterations/manual186_replay
python3 viewer/video_import/fit_dataset_grip.py \
  --dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3 \
  --result YOUR_FRESH_STAGED_RESULT --manual-correction-frames 185
```

单帧报告在 `frame186_manual_correction/manual_correction_report.json`，全段对照在 `manual_parameter_optimization/manual_comparison.json`。独立 [原因／源图对照](http://127.0.0.1:18769/datasets/3aa5d917272490566237eaff8d1a684e/result/viewer.html) 与 [第186帧三维候选](http://127.0.0.1:18769/datasets/3aa5d917272490566237eaff8d1a684e/result/edit3d.html) 已生成，只有zero185球拍与原v9不同；人体和所有其他帧保留。review_manifest记录页面／数据／模型SHA；浏览器已核查186/249帧、人工叠加、握拍视角及无控制台错误，截图 `frame186_manual_correction/browser_preview186.png`。31项相关测试、编译检查和Graphify更新完成，主v9三个保护文件SHA不变。本轮仅本地CPU诊断，没有新增GPU运行、没有替换主预览。下一步优先核查柄末端／喉部物理定义和冻结握点的投影，再用可信镜中配对评估相机／镜面；之后重生成PTS对齐自动观测，采用明确训练／回归／独立验收分割比较。

#### 实际握点／柄底／拍喉定义核查及镜面内参敏感性（2026-10-02）

新增只读 `audit_racket_physical_definitions.py`，结果保存于 `output/offline_iterations/20261002_joint_evidence_review/physical_definitions_v1/`。报告绑定归一化视频SHA、v9实际人体candidate、主v9球拍、旧10／新6人工标签、v9 staged镜面与v6原生相机的SHA。v9显示网格使用既有source_roots重基准，不能直接要求candidate_root等于mesh_meta；审计验证全部249帧世界网格最大差5.96e-8m、预览manifest SHA和全部球拍旋转／平移一致。

物理定义发现：

- `handle_end`模型坐标为(0,0,0)，应对应柄底端面轴心；可见边缘、手腕、握柄中点不可互换。
- `grip_contact`使用MCP/PIP走廊并偏移13mm，当前球拍握点又固定为柄底以上45mm。两者都是估计先验，无实测握点／皮肤接触。原资产准备脚本曾使用100mm，方向拟合后固定45mm。
- `prepare_wilson_model.py`把`throat_y_m`设为线床顶点最小y（35.0847cm），实际为线床下缘中心。界面只写“拍喉”，含糊于V形分叉或喉桥。此次记录定义但不重解释／改写历史像素；实际源图是否标同一特征仍需按清晰帧逐点复核。
- 第186帧真人柄尾可见，原人工点在其附近；人工柄底到当前握点37.2423原图px，而模型两点距约15.2px。固定原姿态／深度的射线－轴线最小二乘给出握点距柄底110.2mm、射线间隙7.91mm，仅为模型代理。原v9握点到v9关节MCP/PIP先验差3.94mm，也不能证明手指网格接触。
- 镜中柄底标记在手部／腕附近，原像素不足以确认独立可见端面中心；已询问用户是可见柄底还是估计点，尚未收到确认。保留原标记，将它视为物理对应待核查，不作为相机真值。

只改变`grip_y_m`，固定当前v9手部握点、镜面、相机、拍形，并分别重新拟合旧8个完整人工帧的真人旋转。距离扫描4.5／6.5／8／10／12cm；每帧均拟合本身人工点，不能当作保留帧准确率，新6帧未进入此扫描。第186帧对照（原图px）：

| 假设距柄底 | 真人五点RMS | 真人中位 | 柄底误差 | 镜中五点中位 |
|---|---:|---:|---:|---:|
| 4.5cm | 10.0901 | 3.0690 | 21.9092 | 32.5365 |
| 8cm | 4.6345 | 3.9834 | 7.5902 | 37.1168 |
| 10cm | 5.3361 | 5.3176 | 2.2868 | 44.1652 |

8cm虽改善第186帧整体及柄底，却恶化镜中；UI206真人RMS 5.53→12.77px，其它帧偏好的距离也不同。低中位数可掩盖单个柄底大误差，必须同时列RMS及逐点误差。此次没有采用8／10cm，也没有据此测量握法。

相机敏感性另行使用冻结v6原生人体和v9 staged镜中COCO点，肩／髋／膝／踝199个训练帧、50个保留帧（每5帧取1帧），手腕／手臂仅评估；所有人工球拍点仅评估，完全不训练相机。固定人体及根平移，分别允许镜面法向／距离、再加全局焦距比例、再加主点偏移。诊断范围法向各±5°、距离±2m、焦距1/1.15至1.15倍、主点各画幅±5%；该范围是诊断边界，不是已验证参数。

| 候选 | 焦距比例 | 核心人体保留中位／P95 | 手臂保留中位 | 新审核球拍极线中位 | 原生真人投影变化中位 |
|---|---:|---:|---:|---:|---:|
| 当前／只调镜面 | 1.0000 | 15.1482／65.8020 | 18.8869 | 13.8872 | 0 |
| 镜面＋焦距 | 1.1023 | 12.8874／68.9112 | 24.4708 | 38.0564 | 53.4228 |
| 镜面＋焦距＋主点 | 1.1500 | 12.9562／68.8068 | 23.0770 | 49.9246 | 102.2679 |

单位均为原图px。最后候选焦距触边、主点偏(17.8323,71.2359)px；虽能把第186帧柄底／拍喉极线偏差降到5.39／2.56px，但拍尖仍35.43px、新球拍配对退步。因此全部拒绝推广。人体与COCO都是估计，改焦距可能补偿人体误差；原生真人投影变化是自一致诊断而非独立人体准确率。镜面距离在极线约束中消去，不能通过改距离解决错误配对。镜头畸变、物理镜面与相机均未实测。

```sh
python3 viewer/video_import/audit_racket_physical_definitions.py \
  --dataset output/video_library/85ade7a072984579831f5cb76e8e5fd3 \
  --staged output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/staged/result \
  --native output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v6/native/reconstruction.npz \
  --body-candidate output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/gpu-fit/mhr_refit_candidate.npz \
  --labels output/video_library/85ade7a072984579831f5cb76e8e5fd3/iterations/grip_refit_v9/staged/result/racket_landmarks.json \
  --new-labels output/offline_iterations/20261002_joint_evidence_review/new_label_evaluation/labels_validated.json \
  --output output/offline_iterations/physical_definition_replay
```

报告`physical_definition_report.json`、实际资产定义图、8帧距离敏感性图与真人／镜中原像素裁剪保存原目录，代码snapshot和发布manifest可复核来源。独立 [互动核查页](http://127.0.0.1:18769/datasets/9a483314db103cc89b3e21592651efb9/result/viewer.html) 可切换4.5／8／10cm；浏览器验证8cm状态数值、图像切换及无控制台错误，截图`browser_grip8.png`。9项相关测试通过，包括已知几何的握点距离反解、镜面距离的极线不变性，以及修改人体保留／人工球拍点不会改变相机训练解。主v9保护文件SHA不变，无GPU运行或新验收发布。

下一步依据用户确认的可见性，优先统一端面轴心与线床下缘特征，保留遮挡／不确定点；握点距离需要清晰手部证据或实测，不直接全段改成8cm。现有相机／镜面继续保留，可信配对不足时使用已知尺寸的真人／镜中固定几何核查内参和镜面，再评估手部深度与联合拟合。


#### 人工握点标注与图像校准（2026-10-02）

新增 `grip_center` 原图像素及逐观测 `grip_confirmed.points / mirror_points`。握点定义为手掌包握区域中心在拍柄轴上的位置，不等同于手腕或柄底端面轴心；旧 `throat` 界面名称明确为线床下缘中心，历史坐标不重解释。未确认的握点只保存，CPU／GPU残差、距离估计均排除；真人和镜中分别确认。原5点、视频SHA、尺寸、fps与旧schema兼容。

当前主视频已安装 [握点标注页](http://127.0.0.1:18769/datasets/85ade7a072984579831f5cb76e8e5fd3/result/grip_annotation.html)，默认第186帧。操作：真人／镜中切换 → 点击清晰包握区域 → 勾选确认 → 保存握点并校准。方向键移动1原图像素，Shift为10像素；遮挡留空。未来打包会自动附带同一页面。当前其它审计数据集缺少完整主球拍／元数据，不声称已安装全库。

新增 `racket_grip_calibration.py` 和 `/api/videos/<id>/racket-grip-calibration`。同帧5个真人球拍点先独立估计刚体位姿，再从已确认握点估计柄底沿轴距离，使用原图分辨率与现有相机／拍形。默认距离搜索1–20cm，拒绝边界解，握点残差阈值4 canonical px、五点RMS阈值8 canonical px，多帧距离相对中位最大差须≤2.5cm。单帧仅形成图像候选；厘米值不是实测，镜面不用于距离估计。握点射线采用原握点深度，只调整候选球拍，不改变人体；手指皮肤接触与时序仍未验收。

保存分别写 `racket_landmarks.json`、`racket_grip_calibration.json` 和 `racket_poses_grip_calibrated.json`，最后两者明确 `accepted=false`。报告记录逐帧残差、样本差异、训练编辑角色和来源SHA；主 `mesh_refined.bin`、`mesh_meta.json`、`racket_poses.json` 保持不变。新标注页和主Viewer使用标签SHA拒绝已知并发修改。计算先完成再保存，求解错误保留原标注。主Viewer“版本”可选择人工握点候选；检查主人体／球拍SHA，来源改变需重新校准。联合预览继续禁用属于旧人体的方向候选；发布脚本保留带此保护的校准控件。

全段CPU拟合可读取握点距离和确认射线；`--grip-calibration-heldout` 接受zero-based验证帧，排除其距离及握点射线校准。GPU观测新增握点残差，保留既有训练／验证划分；preflight发现距离校准帧与GPU保留帧重叠时阻止运行，要求仅用训练帧重校准。CPU旧人工点流程本身仍是训练编辑，不能把该选项解释为完整独立验收。本轮未运行新的真实视频全段拟合或GPU任务。

32项相关测试通过，包括已知几何9.5cm恢复、确认过滤、逐视图权重、保留帧隔离、错误视频与并发修改拒绝、主几何SHA保护、实际HTTP处理器以及CPU拟合集成。浏览器使用隔离副本 `3f5ecf0f7dd541ed9fbe6b9ba2705b74`，合成5点与已知9.5cm握点完成点击→确认→POST→重新加载；反解9.506cm（点击像素取整），与实际视频的人工证据无关。另验证主Viewer候选切换、旧人体方向候选禁用及控制台无错误。测试副本完成后移出视频库，报告留在 `output/offline_iterations/20261002_joint_evidence_review/grip_manual_feature/`。

本地API使用既有cloud配置恢复18768服务；仅重建18769 Viewer，无GPU调度。主视频标签未替用户新增握点；下一步需用户标清晰握点，再查看图像候选、逐点RMS和镜中投影，最后按训练／保留分割验证，不能只按同帧贴合判断通过。


#### 用户握点已收到：确认状态待核查（2026-10-02）

主视频实际保存32个真人握点（UI186、187、190–215、246–249）和11个镜中握点（UI187、196–201、246–249），所有 `grip_confirmed` 为false。正式校准报告 `no_usable_confirmed_grip`，没有生效距离；已询问用户哪些标记清晰可见，不代替用户把false改成true。仅UI186／206同时具备5个完整真人人工球拍点；其他握点可提供连续接触观测，但不能各自单独反解距离。

只读报告 `output/offline_iterations/20261002_joint_evidence_review/manual_grip_received_v1/grip_review_report.json` 绑定当前标签、主v9、镜面和源视频SHA；保存原标签快照、审计代码snapshot与5帧原像素裁剪。对尚未确认的坐标作诊断，主v9握点投影与真人标记差中位8.39、P95 20.86原图px；镜中中位13.54、P95 38.36px。11组真人／镜中配对极线误差中位12.07px，UI187最大54.09px；尚不能作为镜面／相机正确性证明。

另在离线副本假设这些标记已确认，以UI186／206训练点做条件反解：分别9.0988／10.2694cm，统一中位9.6841cm，离中值最大0.5853cm。两帧独立位姿五点RMS 1.57／2.28px，握点沿轴残差0.88／1.59px。相机、拍形和握点深度仍为假设，厘米值不是实测；这不是独立验证或正式校准。

该条件候选固定原握点深度、使用人工射线和统一距离，仅改变两个球拍帧：UI186真人五点RMS 24.46→4.42px，但镜中五点38.36→43.33px；UI206真人10.76→4.56px，镜中三点16.56→16.33px。握点射线约束使该点误差为0属于构造结果，不能当准确率。两帧旋转相对原v9变49.06／35.09°，握点相对原先验移5.20／70.62mm；没有修改人体，不能证明手指接触改善。全段最大角步长36.93→66.49°，最大角加速度14.43→95.90°/帧²，拒绝把这个单帧编辑候选替换主版本。

主标签、mesh、meta、球拍、相机及镜面保持收到时状态。后续先获得可见性确认，复核UI187镜中配对，再把连续握点按明确训练／验证分割加入时序与手部联合约束，避免只改两个完整人工帧造成跳变。本次无GPU任务、无新验收版本。


#### 握点确认生效与冻结人体时序对照（2026-10-02）

用户回复“已确认”，按此前问题的全量范围，将32个真人、11个镜中已保存握点记为确认；未改任何像素坐标。通过带原标签SHA的正式API生成距离报告和独立人工握点候选。当前页面刷新可看到已确认状态、9.68cm图像估计和3D候选入口；这两个距离样本来自UI186／206，仍非实测，候选 `accepted=false`。确认前快照、语义记录和API响应在 `manual_grip_confirmed_v1/`，主v9三个保护文件SHA不变。

另完成249帧CPU刚体／握点时序A/B，冻结实际v9人体、相机、镜面；不用有历史PTS问题的自动轮廓，也不把未通过校验的镜面作为训练约束。以实际主v9的R和掌内锚点为先验，联合优化旋转修正与锚点偏移；这不是原生MHR人体联合拟合。实验参数和SHA在 `temporal_ab/protocol.json`，执行代码snapshot和日志在上层目录。两组完全相同，只比较4.5cm旧距离与仅训练帧206估计的10.2694cm。网页9.6841cm是两帧编辑估计，不能直接带入保留186的拟合验证。

训练24个真人握点，8个握点回归帧（UI186、193、198、203、208、211、213、248）不训练；旧UI18／186球拍及新六帧UI31／81／131／156／211／231球拍仅评估。相同图像和标签已在此前诊断看过，仍是固定回归划分，不声称全局未见的独立验收。所有镜中人工点仅评估。

| 原图像素指标 | 主v9 | 同握点训练／4.5cm | 同握点训练／训练距离10.27cm |
|---|---:|---:|---:|
| 8个真人握点回归中位／P95 | 10.68／15.43 | 4.25／20.58 | 4.25／20.42 |
| 旧10个真人球拍点中位／P95 | 15.25／33.85 | 15.25／33.16 | 19.11／42.20 |
| 新28个真人球拍点中位／P95 | 13.68／34.20 | 16.25／37.68 | 9.03／39.88 |
| 新9个镜中球拍点中位／P95 | 33.25／55.37 | 33.04／55.40 | 28.78／48.37 |
| 最大角步长／° | 36.93 | 36.03 | 35.88 |
| 角加速度P95／°每帧² | 9.30 | 9.56 | 9.51 |

两组数值求解收敛，时序校准候选改善新点中位却恶化真人握点P95、旧球拍回归、新球拍P95及加速度P95；不替换主版本。UI175的旋转诊断边界两组均触及，校准组也触及锚点边界；该帧来自此前只有拍尖／两侧的人工点，不能把失败归因于新握点或直接放宽边界。锚点相对原先验最大偏101mm，没有人体皮肤接触验证；源v9的float32旋转正交误差约2.24e-7，候选保持同精度并通过1e-6旋转数值检查。

结果 `temporal_ab/comparison.json` 和 `comparison.png`，判定 `rejected_keep_main`。网页仍可查看人工两帧编辑候选，但时序两组只留隔离结果，没有GPU运行或已验收版本。下一步优先核查握点深度／手部模型和相机镜面对应，再在同冻结划分下使用原生MHR回放做手／拍联合优化；不能只凭真人握点中位改善推广。


### 固定9cm用户参数与249帧回归（2026-10-02）

当前主结果 `85ade7a072984579831f5cb76e8e5fd3/result/racket_grip_settings.json` 明确source=user_specified、grip_from_butt_m=0.09及视频身份。人工校准保存、完整CPU拟合均优先采用该值；estimated_grip_from_butt_m仍为9.68cm诊断。两种值不可混用为实测。

产物：`output/offline_iterations/20261002_joint_evidence_review/fixed_grip_9cm_v1/`。阅读REPORT.md；protocol/标签快照/对照/焦点帧/代码快照/发布哈希/备份均保存。主默认固定9cm保留v9人体旋转与握点；`racket_poses_grip_calibrated.json` 是独立9cm时序候选，accepted=false。后续保存人工点会重新生成9cm人工点候选；本次时序结果永久保留在temporal_ab，不依赖该可覆盖文件。

运行实验脚本需要更改至新的输出目录且使用对应4.5cm输入备份；不能直接对已发布9cm主文件声称复现原对照。34项相关测试及浏览器切换通过；主poseSHA d7ec5e88457bbd9d9009830816b728c5886f4ae2097081b780791691d8f08dc6。


### 全自动握持距离只读核查（2026-10-02）

工具 `viewer/video_import/audit_automatic_grip.py --help`。使用当前record指定attempt原始reconstruction.npz手关节，拍长来自racket_dimensions.json，YOLO权重与准确PTS绑定原视频。predict阶段没有手工标签／主球拍／固定9cm输入；冻结predictions.json后evaluate阶段读取人工点。手关节轴线法与等深像素比例法独立输出，拒绝项保留原始值及原因但不计入有效距离。

本次产物 `output/offline_iterations/20261002_joint_evidence_review/automatic_grip_validation_v1/`；完整复现命令、代码及输入哈希见REPORT.md。CLI主标签评估与本次补充6帧合并评估的范围不同，切勿混称相同结果。实际预测快照与后续评估代码分别保存；精确复现需对应代码版本和冻结源文件。23项相关测试通过。新增核查页：http://127.0.0.1:18769/datasets/85ade7a072984579831f5cb76e8e5fd3/result/automatic_grip_validation.html 。结论未验收，保留主固定9cm；无GPU人体重拟合。


### 其余8段固定9cm应用（2026-10-02）

入口 `apply_grip_collection.py --help`，发布总览 `publish_grip_collection_review.py --help`；实际命令及代码快照见 `output/offline_iterations/20261002_grip_collection_9cm_v1/REPORT.md` 与 `reproduction.json`。48.53起8段使用各自 `record.attempt` 原生archive、归一化SHA与原视频精确PTS；复制mutable JSON、只硬链接不可变大文件，固定9cm，手部先验不可靠时降权。先生成独立自由测距预测，再做真人候选角色过滤与整段刚体时序拟合；镜中只展示／诊断。

现有输出目录已完成，重复执行跳过有publication.json的段，不代表重新计算；中断stage要求先核查日志。复现应在新的输出目录及隔离视频库运行，避免覆盖现有候选或已验收48.43。报告发布代码的当前版本独立保存于 `report_code_snapshot/`。各段发布manifest保护原人体、视频、纹理；全局最终验证保护48.43验收SHA。

8/8段共1993帧（4帧隐藏），均完整播放至终点并抽查五视角；全部保留未验收。数值自一致门限与皮肤接触代理不能作为独立准确率。`automatic_distance_validation_queue.json` 冻结53帧，只供独立参考评估；原图标签必须明确握点／柄底／拍尖可见性，厘米准确性还需独立实测。固定9cm不是实测真值，不得回流为自动测距评估答案。相关10项测试及148个HTTP链接通过，Graphify按源码排除output重建。
