# Local video import and generation

For end-to-end reproducible commands, environment/model identities, input/output
contracts, recovery and backup, start with
[REPRODUCTION_RUNBOOK.md](REPRODUCTION_RUNBOOK.md). The observed environment and
historical artifact digests are in [REPRODUCTION_BASELINE.json](REPRODUCTION_BASELINE.json).

Cloud GPU deployment components, execution contracts, generated-data inventory,
and per-run records are tracked in [CLOUD_GPU_PLAN.md](CLOUD_GPU_PLAN.md).

MHR joint-fit execution, acceptance, outstanding stages and the next capture
budget are recorded in [MHR_FIT_PROGRESS.md](MHR_FIT_PROGRESS.md). Each
`run_fullbody_refit.py` attempt automatically writes `completion.json` and
content-addressed events under `output/mhr_fit_progress/`; completed GPU
optimisation is recorded separately from an accepted fit. The existing nine
videos were analyzed across all 511 subsets; retaining their surface support
does not establish accurate body or hand/racket fitting.

Default Viewer via local library: http://127.0.0.1:18768/default/viewer.html

The library serves the existing default dataset directly (including HTTP Range),
so its default-video link does not depend on the Docker service on port 18766.
Library: http://127.0.0.1:18768/import.html

Start the companion service on the Mac (nginx stays a static Docker deployment):

```sh
python3 viewer/video_import/server.py
```

Uploads are stored under ignored `output/video_library/<random id>/`. Imported
videos are normalized to 25 fps, max width 2560, H.264 and no audio. The original
upload is retained privately as original.video and is never served over HTTP. Limits: 1 GB and 120 seconds. The current
sample dataset is never changed or automatically replaced. Video library access
is bound to loopback and checks Host and Origin. It is not a public upload API.

## Real generation prerequisites

The built-in runner requires an already installed SAM 3D Body environment and
locally provisioned model assets. Set:

- `SAM3D_BODY_CODE`: SAM source directory
- `SAM3D_WEIGHTS`: directory with model.ckpt, model_config.yaml and assets/mhr_model.pt
- `SAM3D_FACES`: optional .npy triangle topology; defaults to the SAM estimator topology
- `VIEWER_SEGMENTATION_WEIGHTS`: local Ultralytics person segmentation weights

Use the Python environment containing torch, opencv, numpy, ultralytics and SAM
dependencies to start the service. It automatically enables the built-in runner
when these paths exist. No model weights are downloaded automatically. Without
these prerequisites upload/preview remains available and generation is disabled.
Restart the service after configuring models. Imported videos survive restart.

The built-in runner chooses the largest initial person and follows box overlap;
it fails explicitly on loss of tracking. Manual multi-person target selection remains future work. A separate multi-video MHR UV fusion candidate and review Viewer are available in [MULTIVIDEO_TEXTURE_ITERATION.md](MULTIVIDEO_TEXTURE_ITERATION.md); imported Viewers still use source-video projection by default.
The cloud multiview runner now adds independent mirror SAM3D and dual-object SAM2;
local postprocessing supplies racket fitting and per-video mirror calibration. No copied old calibration is used.

A trusted local adapter, including one that dispatches to an existing GPU worker,
can be selected with `VIEWER_GENERATOR_COMMAND`, a JSON array of command arguments.
The service appends `--video /absolute/source.mp4 --output /absolute/work`.
HTTP clients cannot configure this command. Adapter exits nonzero on failure.
It exports `reconstruction.npz` with no pickled objects:

| Key | Shape / meaning |
| --- | --- |
| vertices | float [F,V,3], camera axes, before camera translation |
| faces | integer [T,3], matching vertex indices |
| source_roots | float [F,3], camera translation |
| focal | positive float [F], normalized video's pixel coordinate system |
| masks | uint8 [F,H,W], person confidence 0–255, full-image coordinates |

F must exactly match the normalized input video. Results are validated and
published only after packaging succeeds; jobs are serialized, have a four-hour
timeout, and interrupted jobs become retryable failures on restart. Errors are
shown in the library; detailed runner logs stay in attempts/<number>/generation.log; remote logs, hashes and manifests stay in its work/ folder.
Do not reuse the default video's output as a generator for a new video.

```sh
python3 -m unittest discover -s viewer/video_import -p 'test_*.py' -v
```

Tests use generated color videos and synthetic geometry, not model inference.

## Existing cloud GPU

Use `deploy/3dpose/cloud_gpu/deploy.sh` to provision the worker scripts into an existing GPU Python environment, then `start_local.py` to connect the local library. See the linked Chinese runbook for configuration, real-video smoke commands, records and limitations.

## Mirror markers and Wilson racket (new videos)

The dataset Viewer now has a collapsible **镜面与球拍** panel. Use the same physical corner convention as the default Viewer: ground A/B/C/D, then mirror A′/B′/C′/D′, with A↔A′ etc. Markers support undo, keyboard nudging, perspective grids and JSON import/export. They are saved to `result/calibration_annotations.json`, checked against the video's SHA-256 and image size. Saving changed markers disables the old mirror fit until explicitly recalculated.

Enter measured AB/AD lengths. The 3.3/4.8 values are editable starting values, not measurements of each new video. **拟合镜面** fits the paired corners using the current video focal estimate, checks residuals and positive depths, matches reflected-body projections against current-video person masks, and writes mirror plane/masks. The fit is an estimate; matching corners alone cannot validate body scale or true camera calibration. The new camera clip now also has an automatic joint-based mirror estimate, following the default Viewer's SAM-camera registration approach; paired corner markers remain available for manual review.

Wilson mesh geometry is reused from the default asset; its old motion is never reused. Current-video racket silhouettes are associated with the selected SAM wrist and fitted to a rigid racket. Only bracketed gaps up to 0.6 s are completed by constraints; unsupported frames hide the racket. The planar orientation and real grip/size remain ambiguous. The fitter uses MCP/PIP grasp corridors, wrist anchoring, real/mirror silhouettes and SO(3) temporal constraints.
The dual-view hand display is a bounded preview, not anatomical grip verification or MHR pose refitting.

```sh
python3 deploy/3dpose/cloud_gpu/start_local.py \
  --config deploy/3dpose/cloud_gpu/host.local.json \
  --racket-model /absolute/path/yolo26s-seg.pt \
  --mirror-pose-model /absolute/path/yolo26m-pose.pt

# Enrich an existing ready dataset; the NPZ must include SAM joints.
python3 viewer/video_import/enrich_result.py \
  --dataset output/video_library/DATASET_ID \
  --model /absolute/path/yolo26s-seg.pt --device mps --hand right
```

Current Mac postprocessing uses MPS locally after cloud body inference; `VIEWER_RACKET_DEVICE=cpu` or `cuda` selects another provisioned device. Prerequisites: Ultralytics, PyTorch, OpenCV, SciPy, local segmentation weights and the existing `output/sam3d_cloud/wilson_mesh.bin` / `wilson_model.json` assets. The optional postprocessing is enabled by `--racket-model` / `VIEWER_RACKET_MODEL`; failures preserve logs and fail the combined job for retry. Without it, body generation remains available and the panel reports missing racket observations.

Extra outputs: `racket_poses.json`, `wilson_mesh.bin`, `wilson_model.json`, `person_candidates.json`, `enrichment_manifest.json`. After manual mirror fitting: `paired_ground_calibration.json`, `ground_calibration.json`, `mirror_ground_grid.json`, `mirror_calibration_report.json`, updated mirror geometry/masks. These belong to each dataset, not the default-video folder.

### Automatic mirror registration

`estimate_mirror.py` detects the reflected person's COCO joints, swaps anatomical left/right correspondences, fits a single mirror plane in the current SAM camera frame, and excludes every fifth frame from fitting. Shoulders/hips/knees/ankles determine the plane; arm joints remain in the error report. Texture acceptance uses held-out joint median and independently detected reflected-body box overlap, then the renderer applies mask/depth checks per pixel. Large individual joint errors remain reported; they do not blank a whole frame's texture.

```sh
python3 viewer/video_import/estimate_mirror.py \
  --dataset output/video_library/DATASET_ID \
  --model /absolute/path/yolo26m-pose.pt --device mps
# Add --reuse-observations for a matching existing mirror_pose.json cache.
```

Without adequate evidence, the automatic stage records needs_review and publishes the available body/racket; manual paired calibration remains available. It never supplies default-video mirror poses to another video. Outputs: mirror_pose.json, mirror_joint_fit_report.json, mirror_calibration_report.json and updated mirror plane/masks. The plane and model scale remain monocular estimates, not measured camera geometry.

The local Docker release is documented in ../../deploy/3dpose/README.md and uses port 18769. Its UI and assets are in the image; videos are mounted read-only and the Mac service on 18768 performs writes and MPS/GPU job dispatch.

### 新视频的握持与拍面

启用球拍权重时，自动运行 `fit_dataset_grip.py`。方法沿用默认 demo 的 MCP/PIP 掌内握点和有方向杆轴，加上真人/镜中拍框与整段时序旋转约束。可靠观测不足的原隐藏帧仍隐藏；约束估计以橙色显示。源视频哈希、原 wrist 拟合、检测候选、握持先验和拟合记录保存在本视频结果目录；不读取 demo 的姿态序列。当前只支持右手握持，不能确认握柄棱位或真实手指接触。云 GPU 部署与生成步骤见 CLOUD_GPU_PLAN.md；本机握持后处理无需重新调用 SAM3D。

### 拍面抖动与处理分辨率

导入视频现在保留宽度至 2560 像素、25 fps 的分析/播放视频，并在私有 `original.video` 保留上传原文件。不是把低分辨率视频放大成 4K。SAM3D 使用统一视频，球拍额外读取原分辨率手部/镜中手部局部裁剪（模型输入 960），所有观测回到同一图像坐标系。像素残差使用 1280 宽的规范坐标以保持跨分辨率阈值一致。

握拍拟合降低异常掌部方向对真实拍框的覆盖，限制掌内位置滤波最多偏移 15 mm；在 SO(3) 上按观测可靠度约束角加速度，保留匀速挥拍，补全 ≤0.6 秒且两端有证据的短缺拍。没有证据的长区间/首尾仍隐藏。浏览器按连续视频时间四元数插值，估计色平滑过渡；暂停与拖动精确回到指定帧。

复现/验收命令（当前视频目标，无缺拍；其它存在真实长遮挡的视频应设置合理的 `--max-hidden`）：

```bash
python3 -B viewer/video_import/audit_racket_motion.py \
  output/video_library/85ade7a072984579831f5cb76e8e5fd3/result/racket_poses.json --check
```

输出最大帧间旋转、角加速度和隐藏数；这类稳定性指标不代表拍面三维准确率。原图、局部候选、未平滑基线、握持/稳定性清单、每次推理及发布前结果均按视频/attempt 保留。云推理后按完整链自动运行；`upgrade_resolution.py` 只用于同时间轴的分辨率转移，不是重新 SAM 推理。

## Independent mirror SAM3D and genuine SAM2

See [MULTIVIEW_GPU_ITERATION.md](MULTIVIEW_GPU_ITERATION.md) for deployed components,
commands, NPZ fields, actual run hashes and acceptance metrics. With
`VIEWER_MULTIVIEW=1`, new videos automatically run both stages. R/G mask atlas
channels contain real/mirror SAM2 video propagation; YOLO only supplies prompts.
The default demo remains available. Changed mirror annotations invalidate the
previous hand constraints; recalibration recomputes matching constraints and racket fit.

## 拍柄关键点与方向校准迭代

当前自动几何关键点、镜面射线、留出验证、稀疏校准 API 和实际限制见
[RACKET_DIRECTION_ITERATION.md](RACKET_DIRECTION_ITERATION.md)。

## 多视频纹理与握拍复核

实际运行、数据目录、合成方法和未通过的重拟合候选见 [MULTIVIDEO_TEXTURE_ITERATION.md](MULTIVIDEO_TEXTURE_ITERATION.md)。
