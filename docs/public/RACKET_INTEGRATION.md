# Tennis racket integration for the SAM 3D Viewer

## Current state

The video is 1280×720 at 25 fps for 250 frames. SAM 3D Body supplies the athlete's body and anatomical right wrist (MHR70 index 41), but no racket geometry. The body mask is not a racket mask. The Viewer now includes a manual keyframe editor that records original-video pixel coordinates and exports `racket_annotations.json`.

An exploratory local pass with the available `yolo26s-seg.pt` tennis-racket class did not reliably detect the small, dark racket on full frames. Wrist-centred crops produced some candidates but missed sample frames 0 and 240 at confidence 0.08 and included possible mirror/background detections. This five-frame check is diagnostic, not a measured accuracy rate. Do not feed its boxes directly into 3D reconstruction without review.

## Coordinate and data contract

Use zero-based frame indexes shared by `video.mp4` and `temporal_pose.json`. Manual racket points are `[x,y]` in the original 1280×720 frame, never coordinates from a cropped detection image. The editor records `handle_end`, `throat`, `tip`, and `rim_side`; points hidden by the hand or motion blur can be omitted. Each saved frame has a `source` field, so automated results and manual corrections remain distinguishable.

For 3D fitting, work in the original SAM camera frame: X toward image right, Y down, Z forward. Attach the grip with the right-wrist observation only as a *constraint*; the racket can rotate around the hand and the wrist is not the exact handle endpoint. Use `mesh_meta.json` camera roots/focal length for source-camera reprojection. Apply the same camera-to-ground basis and display view as the body only after fitting. Keep the real racket and its mirror reflection as distinct 2D observations of the same object. The fitted mirror plane is an estimate, not an independent 3D measurement.

## Next stages

1. Mark racket keyframes at different swing angles and export the JSON. Include clear and occluded examples.
2. On the GPU, segment and track the **real** racket separately from the person, using reviewed keyframes as seeds. Track the mirror image separately. Save mask confidence and mark absent/ambiguous frames rather than filling them with guesses.
3. Fit an approximate rigid racket (measured length and head dimensions), constrained by grip location, real-image contour, mirror-image contour when valid, and temporal continuity. Report reprojection error and uncertainty per frame. A single visible view cannot always determine racket-plane depth or rotation.
4. Add a separate racket mesh/texture layer and on/off controls to the Viewer. Verify occlusion against the body and mirror before allowing the object into contact-frame and forehand/backhand analysis.

Acceptance should use held-out manually marked frames: compare endpoint reprojection, grip-to-hand distance, temporal jumps, real/mirror consistency, and failure cases. Keep the 2D observations available beside every 3D estimate for review.
