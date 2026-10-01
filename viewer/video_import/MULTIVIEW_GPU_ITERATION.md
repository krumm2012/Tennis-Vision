# 镜中 SAM3D、双人物 SAM2 与握拍约束

## 1. 本轮范围及流程

2026-10-01，在已有 RTX 4090 上完成真实新视频的双人物推理。默认 demo 保持原入口；新视频结果独立发布到本地 Docker 18769。

```text
上传原视频（私有 original.video）
 → 标准化 source.mp4：最大宽度 2560、25 fps、H.264
 → SSH / SCP + Worker：校验源视频 SHA-256
 → 真人 SAM3D：人体、相机、MHR 关节
 → YOLO 提示框：关联真人和镜中人物
 → 镜中独立 SAM3D：全图水平翻转后推理、X 坐标还原
 → SAM2.1 small：双对象提示、视频传播、像素冲突消解
 → NPZ / 清单 / 日志下载并核对哈希
 → 打包 R 真人 / G 镜中 SAM2 遮罩图集
 → 镜面配准：独立留出 2D 关节 + 人物框验证
 → 多视角一致性筛选：保留真人相机根节点，镜中手指局部约束
 → 原分辨率拍框 ROI + 掌内握点 + 双侧轮廓 + SO(3) 时间约束
 → 本地 Three.js Viewer：视频投影纹理、背面反射纹理、握拍预览
```

镜中 SAM3D 与真人是独立推理。翻转镜中图像后使用同一套解剖 MHR 关节编号；还原时将网格、关节和相机根节点 X 取负，2D X 恢复到原画面。不能在此再次交换左右手。原始镜中虚拟相机结果保留在 NPZ；其镜像网格没有直接作为 Viewer 的新人体显示。

## 2. 已部署组件及配置

| 组件 | 实际位置 |
| --- | --- |
| GPU / Python | RTX 4090 / `/root/tennis-sam3d/venv/bin/python` |
| SAM3D 源码 / 权重 | `/root/tennis-sam3d/sam-3d-body` / `/root/tennis-sam3d/weights` |
| SAM2 源码 | `/root/tennis-sam3d/sam2`，已在同一 Python 环境安装 |
| SAM2 权重 | `/root/tennis-sam3d/sam2_work/sam2.1_hiera_small.pt` |
| 人物提示分割 | `/root/tennis-viewer/models/yolo11n-seg.pt` |
| Worker / 任务 | `/root/tennis-viewer/code/` / `/root/tennis-viewer/jobs/<job_id>/` |
| 本地云适配器配置 | `deploy/3dpose/cloud_gpu/host.local.json`，忽略且不提交 |

`worker.py`、`generate_sam.py`、`generate_multiview.py`、`run_records.py` 由部署脚本上传。部署检查 CUDA、SAM3D 模型、SAM2 包和权重是否存在。本轮复用已安装的 PyTorch 2.4.0+cu121 环境，真实推理通过；并非宣称该版本满足最新 SAM2 所有安装要求。新机器安装和许可请按 [SAM2 官方说明](https://github.com/facebookresearch/sam2) 和 [SAM3D 官方说明](https://github.com/facebookresearch/sam-3d-body) 配置兼容环境，部署工具不自动安装模型环境或下载权重。

配置 `env` 增加：

```json
{
  "VIEWER_MULTIVIEW": "1",
  "VIEWER_SAM2_WEIGHTS": "/root/tennis-sam3d/sam2_work/sam2.1_hiera_small.pt"
}
```

原有 SAM3D 和人物模型配置同时保留。`VIEWER_MULTIVIEW=0` 或省略可使用旧单人物流程。SAM2 使用当前视频检测框全集推导固定双人物裁剪区域，保留原视频坐标；每 25 帧窗口自动选择高置信度提示，实际由 SAM2 视频传播生成 confidence masks。不是把 YOLO 多边形文件改名为 SAM2。网络内部仍按各自模型分辨率缩放；输入 2560×1440 不代表神经网络全程处理原生 1440p 像素。

## 3. 执行命令

从仓库根目录执行；运行云任务时避免覆盖远端源码：

```sh
bash deploy/3dpose/cloud_gpu/deploy.sh --config deploy/3dpose/cloud_gpu/host.local.json
python3 -B deploy/3dpose/cloud_gpu/start_local.py \
  --config deploy/3dpose/cloud_gpu/host.local.json \
  --racket-model /Users/krum5539/Downloads/yolo26s-seg.pt \
  --mirror-pose-model /Users/krum5539/Downloads/yolo26m-pose.pt
```

正常从视频库上传/生成会依次运行全部步骤。不需要对每帧手动修补。仅取云数据、不发布网页时：

```sh
python3 -B viewer/video_import/cloud_adapter.py \
  --config deploy/3dpose/cloud_gpu/host.local.json \
  --video /absolute/source.mp4 --output /absolute/work
```

相同视频补齐多视角可加 `--base-job <已就绪的同视频任务ID>`，Worker 强制核对视频和原 NPZ 哈希后复用真人网格，仍运行新增镜中 SAM3D 与 SAM2。新视频不能复用旧视频输出。此轮复用同视频原分辨率真人结果，验证所有 primary vertices/joints/roots/focal/faces 数组逐值一致；本地拍框/2D 镜中观测也复用同源 attempt3 缓存，并记录哈希。

本地分步脚本：

```sh
python3 -B viewer/video_import/estimate_mirror.py --dataset <dataset> --result <staged-result> --model <pose-weights> --device mps
python3 -B viewer/video_import/multiview_constraints.py --dataset <dataset> --result <staged-result>
python3 -B viewer/video_import/racket_observations.py --dataset <dataset> --result <staged-result> --model <seg-weights> --device mps
python3 -B viewer/video_import/fit_dataset_grip.py --dataset <dataset> --result <staged-result>
python3 -B viewer/video_import/audit_racket_motion.py <staged-result>/racket_poses.json --check
```

`package_result.package(video, archive, destination)` 负责分步打包；destination 必须不存在。`record.json` 的 attempt 必须指向本次工作目录，脚本不允许混用其他视频标定。已有同源 2D 镜中观测可用 `--reuse-observations`，同时核对视频和检测模型哈希。

本地 Docker 更新：

```sh
python3 -B deploy/3dpose/prepare_local.py
VIEWER_BASE_IMAGE=nginx:stable-alpine docker compose -p tennis-3dpose-local \
  -f output/3dpose_local/compose.yaml --env-file output/3dpose_local/.env up -d --build
```

## 4. 新增数据及记录

| 输出 | 含义 |
| --- | --- |
| NPZ `mirror_valid` | `[F]`，独立镜中 SAM3D 有效帧 |
| `mirror_vertices / mirror_joints` | `[F,V,3] / [F,J,3]`，原画面虚拟相机局部坐标，米 |
| `mirror_roots / mirror_focal / mirror_joints2d` | `[F,3] / [F] / [F,J,2]`，原画面相机根节点、焦距、像素关节 |
| `masks / masks_mirror_sam2` | uint8 `[F,288,512]`（本视频），真人/镜中 SAM2 置信度 |
| `multiview_manifest.json` | 视频/模型哈希、裁剪、对象ID、提示帧、角色框、时间、mask 诊断 |
| `multiview_constraints.npz` | 小幅融合手部关节、accepted mask、源视频和镜面哈希 |
| `multiview_constraints_report.json` | 身体/手部一致性、根节点差、采用数量和修正幅度 |
| `mesh_refined.bin` | 稳定网格加有界指骨附近变形；显示预览，不是 MHR pose refit |
| `same_video_cache_manifest.json` | 本次同源缓存复用的文件哈希与理由 |

`person_masks_sam2.png` 的 R 通道为真人，G 通道为通过反射投影关联检查的镜中人物。图集本轮 tile512×288，整体8192×4608。手动重新拟合镜面也继续使用实际 SAM2 mask，不用 YOLO 多边形覆盖。标记改动会禁用原镜面和握拍预览；重新标定后重算约束、握拍并刷新页面。

远端任务保存 `source.mp4`、`request.json`、`status.json`、`worker.log`、`run_manifest.json` 和 `work/reconstruction.npz`。本地保存于 `output/video_library/<id>/attempts/0004/`，含远端日志/清单/NPZ。远端清单同时记录基准 Git 提交和实际源码 SHA，避免把未提交源码误认为基准提交内容。原有发布结果归档到本次 `previous-result/`，保留用户角点。

## 5. 本次实际结果

- Dataset：`85ade7a072984579831f5cb76e8e5fd3`，attempt4。
- 原视频编码 2560×1440、50 fps；统一分析输入 2560×1440、25 fps、249 帧。
- 云任务：`15016a8f96d84a50b4cc5fa8a9c454ba`。
- 复用真人任务：`a4cb217fefe14ff4b222f7c2d6bbef73`。
- 源 SHA-256：`725446711d7486ade7dbf05526a9fcac593c9e97e0db4347a551ece19a39e82f`。
- 双视角 NPZ SHA-256：`e54ffcf0d065e3054e2a493910066065809f2bfdf4cf03126539ad68b5678c12`。
- SAM2 checkpoint SHA-256：`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`。
- 镜中独立 SAM3D 245/249 帧；双人物 SAM2 跟踪249帧；反射mask投影门限通过249帧。
- 新增云处理237.30秒（不含已复用的真人推理209.62秒）。
- SAM2 相对 YOLO 提示参考 IoU 中位数：真人0.875、镜中0.834。参考掩码不是人工真值，**不是准确率**。
- 镜面保留已有独立留出 2D 拟合（core中位误差15.15 px），没有将两侧单目整体位置差硬传给真人。
- 两侧身体平移对齐后 core RMS 中位8.41cm；手部局部 RMS 中位3.57cm。189帧手部约束通过，权重最大20%，单关节修正最大约8mm，真人腕和相机根节点不变。
- 球拍67帧轮廓支持、182帧约束估计、0隐藏。上版59/190/0。新增8帧获得轮廓支持，但不代表真实拍面角度已校准。
- 拍面最大单步27.79°，角加速度P95 5.22°；上版27.47°/5.11°，没有证据表明本轮显著减小抖动。此轮主要补齐镜中推理与纹理证据。
- 握拍shaft先验夹角中位仍约39.46°，存在手部/拍框方向冲突，必须复核。握柄棱位、拍面有符号朝向、球拍真实尺寸没有验证。

## 6. 验收与后续

```sh
python3 -B -m unittest discover -s viewer/video_import -p 'test_*.py' -v
node viewer/video_import/test_racket_layer.cjs
```

本轮18项Python测试和浏览器插值测试通过，新增覆盖反射可逆性、深度偏差隔离、失败帧不融合、腕锚点不移动、8mm上限、实际SAM2置信度保留及真人R通道不覆盖。真实输出同时检查旋转正交性、抖动门限、帧数及哈希。

仍待补齐：

1. 独立相机内参、镜面和身体尺度标定；完整人体双视角 MHR 参数联合拟合。这轮身体一致性用于质量门限，不能宣称全身已融合。
2. 专用拍框/拍柄关键点与有向拍面观测；处理手部先验冲突。长期遮挡不能靠无限插值隐藏。
3. 人工选主角、侧置镜/多镜/切镜/多人视频的身份约束。当前自动镜中候选采用较高且邻近真人的布局先验；不适用时需要人工标记，不得冒用另一人。
4. 验证实际握柄棱位、掌指接触和球拍尺寸；现有指骨局部网格变形只是预览。
5. 球轨迹、触球时刻、击球阶段及带时间证据的教学建议；这些没有由SAM模型自动完成。

graphify知识图谱的重建命令已按仓库规则尝试；本机缺少graphify模块，未自动安装，图谱未更新。
