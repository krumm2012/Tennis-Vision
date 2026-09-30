# SAM 3D Body Viewer

Tracked sources for the local SAM mesh, video texture, ground calibration and mirror reference viewer. Install them beside generated data:

```sh
python3 scripts/install_sam3d_viewer.py
python3 -m http.server 8767 --bind 127.0.0.1 --directory output/sam3d_cloud
```

Open http://127.0.0.1:8767/viewer.html. Installing replaces the HTML/JS/helper scripts; it does not replace video, pose, calibration, mesh or texture outputs.

The current viewer expects normalized 1280×720 footage and the existing 250-frame data layout. Required local inputs include `video.mp4`, `temporal_pose.json`, `mesh_meta.json`, `mesh_local.bin`, `mesh_smooth.bin`, `mesh_refined.bin`, `mesh_temporal.bin`, `mesh_faces.bin`, `mirror_geometry.json`, `mirror_geometry_frames.json`, `person_masks_sam2.png`, `person_masks_sam2_stats.json`, and `temporal_texture_sam2.bin`. Optional mirror-grid diagnostics remain beside the data. Generated data and model weights are not included in Git.

After changing the SAM2 mask atlas, regenerate `person_masks_sam2_stats.json` with `build_mask_stats.py`. `build_temporal_texture.py` requires NumPy, OpenCV and Numba and generates the corresponding vertex-color cache.

The final view uses negative depth for points toward the observer. Front view is yaw=0; back view is yaw=π. Positive pitch looks down from above. These are fixed scene views and do not follow the athlete when they turn. Source-camera and reflected-camera texture projections use their original conventions.

Playback updates once per decoded video frame. Canvas sizes change only on resize; ground trajectories and same-frame vertex buffers are reused. Mask statistics are precomputed to avoid reading back the full atlas at startup.

Orientation checks: `node viewer/sam3d/test_orientation.cjs`. See `docs/public/VIEWER_ORIENTATION_ASSESSMENT.md` for the diagnosis, limitations and user acceptance record.

## Tennis racket keyframes

Expand “球拍二维关键帧标注” in the Viewer, pause at a clear frame, and mark the handle end, throat, tip, and both opposite rim points. You can also mark the matching racket in the mirror. Save several frames at different swing angles, then export `racket_annotations.json`. The editor stores drafts in this browser's local storage; exporting is required to move annotations to another machine. Imported files must match video `30.56`, 1280×720, 25 fps, and frame indexes 0–249.

These points are in original video pixels. Missing or occluded points may remain unset; the fitter only solves frames with all five real landmarks. Run `python3 viewer/sam3d/fit_racket_pose.py /path/to/racket_annotations.json` and reload the Viewer to render `output/sam3d_cloud/racket_poses.json`, or import that result in the 3D racket panel. NumPy, OpenCV, and SciPy are required for fitting. The Viewer does not silently interpolate racket poses into unobserved frames. See [racket integration](../../docs/public/RACKET_INTEGRATION.md) for coordinate conventions, dimension settings, and quality checks.

The local data includes a provisional frame-120 example for review, not a validated full-video track. It is amber in 3D because the racket-plane orientation remains ambiguous without mirror points.


### 全帧球拍第一版

完整流程与限制见 `docs/public/RACKET_INTEGRATION.md`。已有候选检测数据时，使用隔离的 SAM2 运行环境：

```sh
output/racket_runtime/bin/python viewer/sam3d/run_racket_tracking.py
python3 viewer/sam3d/fit_racket_video.py
python3 viewer/sam3d/audit_racket_video.py
python3 scripts/install_sam3d_viewer.py
```

跟踪默认使用 CPU，按32帧保存断点；本机 MPS 路径曾产生大量非标注帧空遮罩，不应把正常退出视为成功。输出区分轮廓拟合与插值，单目拍面方向仍有歧义。检查图位于 `output/sam3d_cloud/racket_review/`。Viewer 的“加载服务器最新版”用于替换浏览器先前导入的旧球拍结果。

### 实拍与俯视三维视频对比

从仓库根目录运行 `python3 viewer/sam3d/export_pose_comparison.py`，输出 `output/sam3d_cloud/real_vs_3d_overhead.mp4`。导出以原视频 25 fps 逐帧同步：左侧为原视频，右侧是按地面标定轴垂直俯视的 SAM 人体网格和球拍。俯视镜头跟随人物，网格每格 0.5 米；绿色球拍有轮廓观测，橙色球拍由邻帧插值。球拍作为诊断线框绘在网格上方，因为单目深度和遮挡关系尚未验证。运行需要 NumPy、OpenCV、Numba 和 FFmpeg。

### Wilson 掌柄约束实验版

依次运行 `prepare_wilson_model.py`、`fit_wilson_sequence.py`、`audit_wilson_upgrade.py`（均在viewer/sam3d中）。前者读取Downloads中的Wilson小版GLB；后两者读取output/sam3d_cloud，保留racket_poses_v1.json并输出racket_poses_v2.json。验证后将v2复制为racket_poses.json并运行安装脚本；Viewer提供上一版对比按钮。模型长度及掌内握点仍为待确认初值，具体限制见WILSON_MODEL_UPGRADE_ASSESSMENT.md。

离线对比导出：`python3 viewer/sam3d/export_pose_comparison.py --poses racket_poses_v2.json --out output/sam3d_cloud/real_vs_wilson_v2_overhead.mp4`。

# Three.js renderer migration

The baseline and full joint-fit Viewer use Three.js 0.180.0 for all 3D drawing.
The video annotations and explanatory text remain Canvas 2D overlays.

- `mesh_renderer.js` owns the Three.js renderer, orthographic display camera,
  scene, dynamic body geometry, source-camera depth targets, video and mask
  textures, guides, and coverage sampling target.
- The body uses a custom GLSL material to retain the existing camera-projective
  texture, person-mask rejection, mirror-depth rejection and temporal colors.
  The source-camera passes retain their calibrated perspective projection.
- `racket_renderer.js` registers an independent scene layer. Racket and optional
  hand joints share the body display transform and camera, without patching
  the body renderer or touching native WebGL state.
- Missing body frames preserve the previous complete scene and pause playback.
  The matching video texture is uploaded before the completed scene is drawn.
  Skeleton-only mode does not require a body mesh frame.
- `texture_audit.js` waits for both required mesh arrays before sampling each
  frame, so its coverage report cannot accidentally reuse a previous frame.

## Dependency and build

The checked-in `vendor/` bundle and MIT license are served from the same origin
as the Viewer. There is no runtime CDN dependency. To reproduce the bundle:

```sh
cd viewer/sam3d
npm ci
npm run build:vendor
```

`tools/vendor-entry.js` includes Three.js and its wide-line scene objects.
The package lock fixes the Three.js and esbuild versions. The installer and
Docker packager copy the vendor files into both Viewer routes.

## Verification

```sh
node viewer/sam3d/test_orientation.cjs
node viewer/sam3d/test_mesh_playback.js
node viewer/sam3d/test_playback_state.cjs
node viewer/sam3d/test_guide_buffers.cjs
```

The orientation test executes the actual Three.js camera matrices for 96
combinations and compares them with the established projection and depth
convention. The playback test verifies missing frames are prefetched without
clearing the displayed scene. Browser verification must additionally cover
shader compilation, textured and gray meshes, front/back views, skeleton and
combined modes, racket visibility, video playback/seeking and the 250-frame
texture audit on both baseline and candidate pages.

Regenerate and package with `viewer/sam3d/joint_fit/build_full_viewer.py` and
`deploy/3dpose/prepare.py`. See `deploy/3dpose/README.md` for deployment.

Playback hardening: editing cancels buffered resume, seeking retains the previous
scene until video decoding completes, and failed mesh loading offers a retry.
Guide buffers are reused; coverage sampling runs only while diagnostics are open
or the explicit texture audit is active. Narrow layouts keep transport above the
views. Network recovery and browser compatibility still require browser testing.

## Conservative texture edges

The default edge repair searches at most one mask texel in each direction. It
samples color at a high-confidence mask location only when source depth agrees;
it retains the original projected visibility test and caps recovered confidence.
The advanced setting can disable repair for A/B comparison. Unobserved surfaces
remain gray: reducing gray weight does not establish texture accuracy.

`mesh_meta.json` may declare `image_size: [width, height]` (the coordinate system
of focal lengths, not necessarily encoded video dimensions) and
`mask_atlas_grid: [columns, rows]`. Tile resolution is derived from the atlas.
Legacy defaults are 1280x720 and 16x16. Mirror projection uses image bounds,
depth and the mirror mask instead of a clip-specific rectangular region.
New videos still require their own matched geometry, camera, masks and mirror
calibration; the existing preprocessing and 2D annotation tools are not yet a
general video import pipeline. Candidate temporal texture remains disabled
because its cache is empty and its vertex correspondence has not been validated.

Run `node viewer/sam3d/test_texture_layout.cjs` for metadata validation.

Candidate back-view fix: mirror observations are enabled by default and the
experimental mirror checkbox remains visible. Source-camera depth and mirror
person masks still gate every projected sample. The mirror calibration is an
estimate; remaining holes or misalignment are not validated by coverage alone.
Regression: `node viewer/sam3d/test_candidate_texture.cjs`.

## Evidence-linked coaching

The candidate Viewer includes a reviewed-in-conversation draft (not an automated
model run), six-phase navigation for two swings and an incomplete final swing.
Phase buttons seek both video and mesh; selected clips can pause at their end.
Only one training cue is highlighted, with evidence and uncertainty visible.
`coaching_report.json` is matched to `mesh_meta.json.video_sha256`; new videos
without a matching report show an empty analysis state. Future analysis adapters
must produce their own per-video report. Run `node viewer/sam3d/test_coaching.cjs`.
