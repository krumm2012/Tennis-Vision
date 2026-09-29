# Tennis racket integration for the SAM 3D Viewer

## Current state

The video is 1280×720 at 25 fps for 250 frames. SAM 3D Body supplies the athlete's body and anatomical right wrist (MHR70 index 41), but no racket geometry. The body mask is not a racket mask. The Viewer records real and optional mirror racket points, and can draw a fitted 3D racket in the same WebGL scene and depth buffer as the body. The racket is a dimensioned rigid geometry; a licensed GLB can replace its appearance later without changing the fitted camera transform.

An exploratory local pass with the available `yolo26s-seg.pt` tennis-racket class did not reliably detect the small, dark racket on full frames. Wrist-centred crops produced some candidates but missed sample frames 0 and 240 at confidence 0.08 and included possible mirror/background detections. This five-frame check is diagnostic, not a measured accuracy rate. Do not feed its boxes directly into 3D reconstruction without review.

## Coordinate and data contract

Use zero-based frame indexes shared by `video.mp4` and `temporal_pose.json`. Manual racket points are `[x,y]` in the original 1280×720 frame, never coordinates from a cropped detection image. The editor records `handle_end`, `throat`, `tip`, `rim_side` (frame A) and `rim_opposite` (frame B). A and B must consistently denote the same physical sides in the real and mirror views. Hidden points can be omitted, but an incomplete real set is **not** fitted. Each saved frame has a `source` field, so automated results and manual corrections remain distinguishable.

For 3D fitting, work in the original SAM camera frame: X toward image right, Y down, Z forward. The model's local origin is the handle butt; local Y points toward the tip; local X points toward rim A. Its default dimensions are 0.685 m total length and 0.270 m head width. Measure the physical racket and pass its dimensions to the fitter for better metric scale. The SAM right wrist is a soft grip constraint, not the exact handle endpoint. `mesh_meta.json` supplies the camera root and focal length; the per-frame pose is rotation plus translation in camera metres. The display layer applies the same ground basis, center shift, and orbit view as the SAM mesh. The mirror annotations project the same racket reflected across `mirror_geometry.json`; this plane remains an estimate rather than an independent 3D measurement.

## Fit and inspect

1. In the Viewer, pause on a clear frame and annotate all five real racket points. Optionally switch to **镜中球拍** and mark the same physical landmarks in the reflection. Save and export `racket_annotations.json`. Mark multiple frames at different swing angles.
2. From the repository root, run:

   ```sh
   python3 viewer/sam3d/fit_racket_pose.py /path/to/racket_annotations.json \
     --out output/sam3d_cloud/racket_poses.json \
     --length-m 0.685 --head-half-width-m 0.135
   python3 scripts/install_sam3d_viewer.py
   ```

   The fitter needs NumPy, OpenCV, and SciPy. It rejects missing points, excessive 2D reprojection error (>8 px), large wrist gap (>0.35 m), and large mirror reprojection error (>20 px). It reports ambiguity when two different racket-plane orientations remain similarly plausible. These are review thresholds, not measured accuracy guarantees.
3. Reload the local Viewer or use **导入拟合 JSON**. On fitted frames, compare the colored video landmarks with the projected crosses and check the 3D racket while orbiting the camera. The panel reports pixel reprojection error, grip-to-wrist distance, and ambiguity. Frames without a fitted pose remain empty.

The Viewer can load `racket_poses.json` next to `viewer.html`; Docker packaging includes it if present. No racket pose is inferred from body wrist alone. The current renderer uses a dimensioned frame, grip, throat, and strings. A GLB must be measured and mapped to these same local axes/landmarks before replacing it; simply parenting an arbitrary model to the hand would lose the real racket angle.

The local dataset includes one **provisional review example** at zero-based frame 120 (Viewer frame 121), recorded in `viewer/sam3d/examples/racket_annotations_frame120_review.json`. Its manually estimated points give 6.8 px real-image reprojection RMS and two plausible single-view orientations. It is shown in amber and must be corrected before treating it as a reliable pose. This manual example is preserved separately; the automatic video workflow below produces its own explicitly labelled estimates.

## Next stages

1. Measure the racket and replace the default dimensions; review five-point annotations at swing extremes and contact.
2. On the GPU, segment and track the **real** racket separately from the person, using reviewed keyframes as seeds. Track the mirror image separately. Save mask confidence and mark absent/ambiguous frames rather than filling them with guesses.
3. Fit the remaining tracked frames with temporal continuity and held-out annotations. A single visible view can still leave racket-plane depth or rotation ambiguous; the approximate mirror plane may also be insufficient.
4. If using an existing GLB, establish its local grip, throat, tip, and rim anchors and a physical metre scale. Apply the fitted transform to those anchors, then verify body/racket occlusion and mirror consistency before contact-frame analysis.

Acceptance should use held-out manually marked frames: compare endpoint reprojection, grip-to-hand distance, temporal jumps, real/mirror consistency, and failure cases. Keep the 2D observations available beside every 3D estimate for review.


## Automatic video first version (2026-09-29)

The local workflow extracts full-frame YOLO class-38 candidates with a low proposal threshold (0.005), associates them with the anatomical right wrist, and uses stronger candidates (0.10) as SAM2 conditioning masks. Lowering a proposal threshold increases recall; it does not establish accuracy. A frame-62 diagnostic missed the reviewed racket at 0.25 and recovered it at 0.05 (box IoU 0.719 on this one frame).

`track_racket_masks.py` crops the real-player swing region and restores output coordinates to the original image. Run bounded 32-frame CPU chunks with the local SAM2 runtime and merge their ordered `frames` into `racket_tracked_masks.json`. CPU is the default: the tested MPS path produced empty non-conditioning masks even though the process completed. CPU and MPS require float32 memory compatibility because the upstream predictor stores memory as bfloat16. Keep individual chunk files for recovery.

`fit_racket_video.py` fits a projected head ellipse and a soft SAM wrist anchor, then interpolates missing observations and smooths adjacent orientations. Each record distinguishes `silhouette_fitted` from `interpolated`. All records retain monocular plane ambiguity; interpolation and small contour residuals are not independent evidence of accurate 3D depth. The default racket size is unmeasured. Automatic mirror-view tracking is not part of this version.

```sh
output/racket_runtime/bin/python viewer/sam3d/run_racket_tracking.py
python3 viewer/sam3d/fit_racket_video.py
python3 viewer/sam3d/audit_racket_video.py
python3 scripts/install_sam3d_viewer.py
```

The Viewer displays observation/interpolation counts, per-frame contour residual and detection score, an image-space projected racket outline, and a button to jump to review frames. “加载服务器最新版” replaces an older browser-imported pose file with the newly generated dataset. Audit sheets and `racket_review/audit.json` expose every frame and orientation jumps; they measure internal consistency, not ground-truth accuracy.


### First-version validation result

CPU tracking processed250/250frames;245 passed basic mask area/wrist checks (not an accuracy score). The delivered fit has153silhouette observations and97interpolated frames. A symmetric-plane representation fix reduced >45° adjacent rotation jumps from16to8; remaining Viewer frame numbers are32,42,48,59,153,161,164,171. Maximum jump151.5°, median5.5°; maximum grip/wrist gap0.207m. These remaining jumps are flagged, not silently considered correct.

All ten25-frame review sheets were inspected. Some backswing/occlusion frames and the return-to-ready interval around frames60–89 show visible projected-head displacement; an accepted contour fit can still follow an imperfect mask. Thus this is a reviewable first version, not precise racket-face measurement. Next accuracy work needs reviewed real/mirror racket landmarks and measured dimensions, especially at the listed jumps and long interpolated intervals. Five synthetic geometry tests, the existing Viewer orientation regression, syntax checks and browser load/review-jump checks passed. Results are local; no public redeployment was performed for this iteration.
