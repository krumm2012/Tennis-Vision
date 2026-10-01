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
