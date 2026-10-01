# 云 GPU 生成：阶段 0–1 部署与验收

2026-09-30：已在已有 x-gpu RTX 4090 主机完成真实视频闭环。采用 SSH/SCP 私有传输，复用主机已有 Python/CUDA/SAM 环境；本轮没有构建 GPU Docker 镜像，也没有部署对象存储。默认视频保持原入口，新视频发布独立结果。

## 1. 实现流程

```text
import.html 上传
 → 本地 server.py：H.264 / 25 fps / 最大宽度 2560
 → cloud_adapter.py：视频 SHA-256、SSH 上传、提交任务
 → GPU worker.py：记录任务、启动独立进程
 → generate_sam.py：人物分割与跟踪 → SAM 3D Body → 网格/相机/关节/遮罩
 → generate_multiview.py（VIEWER_MULTIVIEW=1）：镜中独立 SAM3D + 双人物 SAM2
 → 结果与日志回传：核对视频和 NPZ 哈希
 → package_result.py：校验帧数/拓扑/数值，生成稳定网格与诊断
 → 独立 Three.js Viewer：源视频投影纹理、原始/稳定对照
```

本地一次处理一个任务。Worker 不开公网 HTTP 端口；SSH 密钥和主机配置由本机维护。服务仅绑定 loopback，不适合作为公网多人上传服务。

## 2. GPU 组件与环境

| 组件 | 本轮位置/用途 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| Python | `/root/tennis-sam3d/venv/bin/python` |
| SAM 源码 | `/root/tennis-sam3d/sam-3d-body` |
| SAM 权重 | `/root/tennis-sam3d/weights` |
| 分割权重 | `/root/tennis-viewer/models/yolo11n-seg.pt` |
| Worker | `/root/tennis-viewer/code/` |
| 任务 | `/root/tennis-viewer/jobs/<remote_job_id>/` |

已有环境使用 PyTorch 2.4.0+cu121、Ultralytics 8.4.7。SAM 权重目录必须包含 `model.ckpt`、`model_config.yaml`、`assets/mhr_model.pt`；拓扑缺省读取 SAM estimator 的 faces，也可设置 `SAM3D_FACES` 提供匹配的 NPY。模型安装和许可参见 [SAM 3D Body 官方仓库](https://github.com/facebookresearch/sam-3d-body)。部署脚本不安装 CUDA、不下载 SAM 权重；需预先配置环境和模型访问。

## 3. 部署和启动命令

以下命令从仓库根目录执行。实际配置 `host.local.json` 已忽略，不提交密钥、密码或主机凭证。

```sh
cp deploy/3dpose/cloud_gpu/host.example.json deploy/3dpose/cloud_gpu/host.local.json
# 编辑 host / port / root / python / env，先完成 SSH 密钥及 known_hosts 配置。
bash deploy/3dpose/cloud_gpu/deploy.sh --config deploy/3dpose/cloud_gpu/host.local.json
python3 deploy/3dpose/cloud_gpu/start_local.py --config deploy/3dpose/cloud_gpu/host.local.json
```

部署会上传 Worker、生成脚本和记录工具，写入代码提交与文件 SHA-256，检查 CUDA 和模型文件。检查失败时返回非零。它复用已有环境，不覆盖模型或删除历史任务。运行任务期间避免重新部署源码。

浏览器入口：[视频库](http://127.0.0.1:18768/import.html)、[默认视频](http://127.0.0.1:18768/default/viewer.html)。已有服务占用端口时，先停止该服务再启动；不要同时开两个同端口实例。

真实视频验收脚本：

```sh
python3 deploy/3dpose/cloud_gpu/smoke_test.py \
  --video /absolute/path/new-video.mp4 \
  --record output/smoke-record.json
```

它通过同一上传 API 导入，提交生成，等待结果并核对非空质量报告，打印 Viewer URL。它会实际使用 GPU；浏览器贴图验收仍需检查结果画面。也可单独调用适配器（只返回 NPZ，不发布网页）：

```sh
python3 viewer/video_import/cloud_adapter.py \
  --config deploy/3dpose/cloud_gpu/host.local.json \
  --video /absolute/path/source.mp4 --output /absolute/path/work
```

## 4. 生成数据及视频纹理

NPZ 禁止 pickle，`F` 等于标准化视频解码帧数，`V/T` 为模型顶点/三角面数：

| 字段 | 形状/意义 |
| --- | --- |
| vertices | `[F,V,3]`，米，人体局部相机坐标 |
| faces | `[T,3]`，固定拓扑整数索引 |
| source_roots | `[F,3]`，米，相机平移 |
| focal | `[F]`，标准化视频像素焦距 |
| masks | `[F,H,W]`，uint8，全画面人物遮罩；单视角旧输出长边 320，双 SAM2 输出长边 512 |
| joints / joints2d | SAM 3D / 2D 关节，供稳定与后续分析使用 |

贴图通过 `vertices + source_roots` 和每帧焦距投影到当前视频画面，再结合人物遮罩与可见性筛选取色。不是逐帧人工修补。遮罩 PNG 的浏览器 R 通道有效（OpenCV 编码前 BGR 索引 2）。原始网格用于源投影，稳定网格用于显示，避免平滑直接改变取色坐标。

发布产物包括 `video.mp4`、原始/稳定网格 BIN、faces BIN、mesh metadata、遮罩图集和统计、`quality_report.json`、`run_manifest.json`、`viewer.html`。`mesh_smooth.bin` 是实际轻量稳定结果；refined 在双视角一致性通过后为稳定网格加手部小幅显示变形，temporal 仍为原始网格副本，不声称实现高级时序重建。新视频时间纹理融合关闭，未拍摄到的表面保留灰色。后续已接入可选 Wilson 球拍后处理及配对镜面标记/拟合；镜面需完成各视频角点确认。球和实测三维场地仍未生成。

## 5. 任务记录及故障复查

```text
output/video_library/<dataset_id>/
  source.mp4 / record.json
  attempts/0001/
    generation.log / run_manifest.json / quality_report.json
    work/
      remote_job.json / request.json
      worker.log / remote_manifest.json / reconstruction.npz
  result/
    viewer.html / video.mp4 / mesh_*.bin / mesh_meta.json
    person_masks_sam2.png / quality_report.json / run_manifest.json
```

每次失败重试增加 attempt，保留旧日志。成功清单记录源视频哈希、NPZ 哈希、任务 ID、时间、代码版本、GPU、模型哈希、逐帧跟踪框和置信度。后续 Worker 同时记录实际部署 Python 文件哈希，区分未提交源码与 Git 基准。质量报告包含帧数、画面内顶点比例、遮罩面积比例和稳定位移。

**画面内顶点比例不是贴图覆盖率，也不是姿态准确率。** 遮罩面积分母是完整视频画面。第一轮未实现可见表面贴图未覆盖率、分阶段性能统计或教学评分。

适配器等待上限 13800 秒，两个 GPU 阶段分别上限 6800 秒，本地任务上限 14400 秒。推理/跟踪失败、文件损坏或视频哈希不符时不会发布结果，日志尽量回传。服务重启把进行中任务标记为可重试失败；远端独立进程可能继续运行。当前没有自动恢复远端任务、取消、关机或清理机制，重新生成前应按保存的 remote_job_id 检查，避免重复计算。

```sh
# 在已登录 GPU 主机中查看指定任务
/root/tennis-sam3d/venv/bin/python /root/tennis-viewer/code/worker.py \
  status --root /root/tennis-viewer --job REMOTE_JOB_ID
# 同目录 worker.log 和 run_manifest.json 可用于定位失败。
```

## 6. 验收与下一阶段

本次真实任务见 [GPU_FIRST_ROUND_ACCEPTANCE.md](GPU_FIRST_ROUND_ACCEPTANCE.md)。本地合同、失败路径及哈希保护测试：

```sh
python3 -B -m unittest discover -s viewer/video_import -p 'test_*.py' -v
```

下一阶段优先：人工选择跟踪主体、切镜/遮挡处理、贴图覆盖诊断与可靠缓存，复核球拍握持/拍面，再加入球/场地和击球分段。教学指导需要证据时间段、数据质量及人工复核入口；当前新视频不会套用默认视频教学报告。

## 7. 2026-10-01 双人物云推理补齐

实际镜中 SAM3D、SAM2 安装/权重路径、执行命令、NPZ 契约及当前验收见
[MULTIVIEW_GPU_ITERATION.md](MULTIVIEW_GPU_ITERATION.md)。原始视频保存为
`original.video`（不对浏览器公开），标准化和推理输入统一为 25 fps、最大宽度 2560。

## 拍柄关键点与方向校准迭代

当前自动几何关键点、镜面射线、留出验证、稀疏校准 API 和实际限制见
[RACKET_DIRECTION_ITERATION.md](RACKET_DIRECTION_ITERATION.md)。
