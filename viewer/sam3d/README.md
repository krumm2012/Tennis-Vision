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

Expand “球拍二维关键帧标注（第一阶段）” in the Viewer, pause at a frame where the racket is visible, and mark the handle end, throat, tip, and an off-axis rim point. Save several frames at different swing angles, then export `racket_annotations.json`. The editor stores drafts in this browser's local storage; exporting is required to move annotations to another machine. Imported files must match video `30.56`, 1280×720, 25 fps, and frame indexes 0–249.

These points are in original video pixels. They are manual 2D observations and are deliberately kept separate from SAM body joints, person masks, mirror calibration, and any future 3D racket estimate. Missing or occluded points may remain unset. The Viewer overlays saved points only on their annotated frames; it does not silently interpolate them into the other frames.
