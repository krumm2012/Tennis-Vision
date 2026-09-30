# 云 GPU 生成与数据记录计划

状态：**方案文档，云 GPU 尚未部署**。本地视频库支持导入、排队、调用受信任的生成命令、校验结果和发布 Viewer；目前模型环境未配置，`GET /api/videos` 返回 `generation_available: false`。默认视频的预生成结果不构成新视频云推理的验收。

## 1. 目标与边界

选取一段人物清晰、10–20 秒的非默认网球视频，从 `import.html` 导入，经过云 GPU 推理后在它自己的 `result/viewer.html` 中显示有纹理的人体。处理失败时保留任务状态和可复查的日志。第二段不同画幅的视频用于验证尺寸和相机参数未被写死。

第一阶段只要求单人 SAM 3D Body 网格、人物遮罩、原视频投影贴图和原始/稳定对照。球拍、网球、镜面、场地标定和教学报告分别作为后续独立产物，不借用默认视频数据。

## 2. 部署组件

| 组件 | 位置与职责 | 当前状态 |
| --- | --- | --- |
| 视频库 API | 本机 `viewer/video_import/server.py`；统一为 25 fps H.264，保存到 `output/video_library/<id>/source.mp4`，排队及发布 | 已实现，绑定 `127.0.0.1:18768` |
| 云适配器 | 本机受信任 CLI；接受 `--video PATH --output DIR`，负责上传、提交、查询、下载和校验云返回文件 | **待实现** |
| 私有对象存储 | 存放标准化视频及云生成文件；只给任务使用短期、最小权限的访问方式 | **待选择和配置** |
| GPU Worker | 固定版本的运行环境、SAM 3D Body 代码与权重、人物分割模型、FFmpeg；运行 `generate_sam.py` 或经验证的等价入口 | 现有推理脚本，**尚未在云 GPU 验证** |
| 本地打包器 | `package_result.py` 校验视频帧数、网格、遮罩、相机参数并发布独立结果 | 已实现；仍需贴图端到端修复 |
| Viewer | Three.js 按源视频帧、投影与遮罩绘制人物 | 默认视频可用；新视频需真实验收 |

可采用支持 GPU 异步任务及持久化模型权重的平台。若选 Modal，可将 GPU Worker 部署为异步函数，将模型权重放在 Volume，视频和结果经过私有对象存储交换；本机适配器负责提交任务并等待结果。平台 API、身份凭证、成本与数据驻留要求确定后才编写对应部署脚本。参考 [Modal GPU](https://modal.com/docs/guide/gpu)、[异步任务](https://modal.com/docs/guide/job-queue)、[Volume](https://modal.com/docs/guide/volumes)。

## 3. 执行入口与脚本清单

### 已有、可在满足模型依赖后执行

在 GPU 环境中准备 `SAM3D_BODY_CODE`、`SAM3D_WEIGHTS`、`SAM3D_FACES`、`VIEWER_SEGMENTATION_WEIGHTS`。权重目录需包含 `model.ckpt` 与 `assets/mhr_model.pt`，拓扑必须与输出顶点对应。安装步骤和模型访问要求以 [SAM 3D Body 官方仓库](https://github.com/facebookresearch/sam-3d-body)为准。

```sh
python3 viewer/video_import/generate_sam.py \
  --video /data/source.mp4 \
  --output /data/work
```

成功时产生 `/data/work/reconstruction.npz`。本地视频库的自动流程由 `server.py` 调用生成命令，随后由 `package_result.py` 处理；单独运行 `generate_sam.py` 不会发布 Viewer。

```sh
python3 viewer/video_import/server.py
curl http://127.0.0.1:18768/api/videos
python3 -m unittest discover -s viewer/video_import -p 'test_*.py' -v
```

`VIEWER_GENERATOR_COMMAND` 是 JSON 字符串数组，由本机运维配置；服务会追加 `--video <本地标准化视频> --output <本地工作目录>`。HTTP 上传者不能选择命令。当前服务一次运行一个生成任务，单任务上限两小时；重启期间的任务转为可重试失败。

### 待实现的云脚本与接口

| 拟新增文件 | 命令/职责 | 完成条件 |
| --- | --- | --- |
| `viewer/video_import/cloud_adapter.py` | `python3 .../cloud_adapter.py --video PATH --output DIR`；上传视频、提交云任务、查询、下载 `reconstruction.npz`，失败时非零退出 | 可由现有 `server.py` 调用，任务 ID 和错误写入日志 |
| `deploy/3dpose/cloud_gpu/worker.py` | 接收私有输入引用，在固定模型环境中运行推理，上传产物及运行清单 | 一段真实视频的帧数、拓扑和遮罩校验通过 |
| `deploy/3dpose/cloud_gpu/deploy.sh` | 构建/部署固定版本 GPU Worker 与模型挂载 | 记录镜像摘要、代码提交、模型版本，不把密钥写入仓库 |
| `deploy/3dpose/cloud_gpu/smoke_test.sh` | 用指定测试视频提交任务、等待结果、检查数据及前后视角 | 从导入到 Viewer 完整通过 |

适配器完成后再设置例如 `VIEWER_GENERATOR_COMMAND='["python3","viewer/video_import/cloud_adapter.py"]'` 并重启本地视频库。**以上云脚本尚不存在，这个配置当前不能直接执行。**

## 4. 生成数据合同

云端第一阶段返回 `reconstruction.npz`，禁止 pickle；`F` 必须与标准化视频的解码帧数一致：

| 字段 | 形状与单位 | 用途 |
| --- | --- | --- |
| `vertices` | `[F,V,3]`，float，人体局部相机坐标，米 | 每帧网格 |
| `faces` | `[T,3]`，整数，索引范围 `[0,V)` | 固定拓扑 |
| `source_roots` | `[F,3]`，float，相机坐标，米 | 将局部网格移回原相机 |
| `focal` | `[F]`，正数，标准化视频像素 | 原视频投影贴图 |
| `masks` | `[F,H,W]`，uint8，0–255，覆盖完整画面；可等比例降采样，不能裁切人物局部 | 人物遮罩 |

现有打包器还生成 `video.mp4`、`mesh_local.bin`、`mesh_faces.bin`、`mesh_meta.json`、`person_masks_sam2.png`、`person_masks_sam2_stats.json`、`temporal_texture_sam2.bin` 与 `viewer.html`。当前新视频的 `mesh_smooth.bin`、`mesh_refined.bin`、`mesh_temporal.bin` 只是原始网格的链接；时间稳定、球拍、镜面和教学数据尚未在通用打包链中生成。

**贴图验收点：** `package_result.py` 使用 OpenCV 的 BGR 数组写 PNG，索引 2 经编码后是浏览器读取的 R 通道，正好供 `mesh_renderer.js` 读取。已用 PNG 解码核对通道转换；仍需对一段真实新视频检查人物表面是否有纹理、遮罩边界是否漏色、背面是否按预期保留灰色。默认视频的镜面数据不允许作为新视频的占位输出。

## 5. 每次运行必须记录

建议在 `output/video_library/<id>/` 保留现有 `record.json`、`generation.log`，并为每次尝试新增 `run_manifest.json` 与 `quality_report.json`。失败尝试也要保留运行摘要。`run_manifest.json` 建议字段：

```json
{
  "schema_version": 1,
  "dataset_id": "任务 ID",
  "attempt": 1,
  "source_sha256": "标准化视频 SHA-256",
  "source_frames": 0,
  "source_fps": 25,
  "source_size": [0, 0],
  "git_commit": "生成代码版本",
  "worker_image_digest": "GPU 镜像摘要",
  "sam_model_id": "模型和权重版本",
  "segmentation_model_id": "遮罩模型版本",
  "gpu_type": "GPU 类型",
  "remote_job_id": "云任务 ID",
  "started_at": "ISO 8601 时间",
  "finished_at": "ISO 8601 时间",
  "status": "ready 或 failed",
  "artifact_sha256": {"reconstruction.npz": "文件 SHA-256"}
}
```

`quality_report.json` 至少包含各阶段耗时、实际处理帧数、跟踪中断数、遮罩覆盖分布、网格无效值、投影是否落在画面内、贴图未覆盖率及失败帧列表。报告中的平滑指标只描述平滑程度，不能标作姿态准确率。视频、模型和报告通过哈希关联；后续教学报告也必须包含对应视频哈希和证据时间段。

日志不写凭证或可长期使用的签名 URL。视频与模型权重分别保存在受限位置；对象存储的过期、删除与重试策略在选定云平台后落实。

## 6. 实施顺序与验收

1. 固定当前工作区版本，验证新视频遮罩通道及帧/分辨率合同；用合成数据和真实短片检查 Viewer。
2. 确认模型访问与 GPU 运行环境，在 GPU Worker 上直接运行 `generate_sam.py`；验证 `reconstruction.npz` 的全部字段。
3. 实现云适配器和私有数据传输，接入 `VIEWER_GENERATOR_COMMAND`；覆盖成功、超时、失败、重试与服务重启。
4. 在两段不同画幅的非默认视频上端到端验收：帧同步、前后视角、遮罩/贴图覆盖、没有默认视频资产混入。
5. 增加球拍、球、场地、击球阶段和教学报告数据；每项提供质量分数和人工复核入口。

验收记录附任务 ID、运行清单、质量报告和 Viewer 地址。只有真实 GPU 任务完成且画面经检查通过，才把“云生成可用”标为完成。
