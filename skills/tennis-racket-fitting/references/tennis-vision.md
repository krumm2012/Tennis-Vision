# Tennis-Vision workflow

Repository: `/Users/krum5539/Documents/Tennis-Vision`; commands run from its root. Use the current dataset record to locate its attempt archive; do not assume attempt 4 for new clips.

## Current modules

- `viewer/sam3d/fit_wilson_grip.py`: demo MCP/PIP grasp corridor and shaft prior.
- `viewer/video_import/racket_observations.py`: native-resolution real/mirror ROI evidence.
- `racket_keypoints.py`: inferred butt/throat/tip/rim points and role association. These are candidates, not ground truth.
- `racket_review_frames.py`: distributed sharp-frame crops and review sheet.
- `racket_calibration.py`: video-scoped measured versus assumed dimensions.
- `fit_dataset_grip.py`: rigid fit, hand evidence, direction and quality gates. Default and directional geometry are separate assets.
- `grip_contact_audit.py`: palm-anchor gap, raw hand/shaft conflict, joint-cylinder proxy.
- `refit_readiness.py`, `refit_fullbody.py`, `run_fullbody_refit.py`: private native-MHR optimization; assumptions/automatic observations require explicit provisional flags.
- `audit_racket_motion.py`, `audit_racket_direction.py`: motion and observation metrics.

## Execution

Read each CLI's `--help` before use; output paths must remain scoped to the target video. GPU host configuration is private at `deploy/3dpose/cloud_gpu/host.local.json`. Deploy versioned sources using `deploy.py`, infer through `cloud_adapter.py`, and keep `remote_job.json`, manifests, logs and verified archive hashes. Do not print credentials.

Local Viewer: `http://127.0.0.1:18769/datasets/DATASET_ID/result/viewer.html`; write API: port 18768. Use the Browser skill for visible UI checks. Docker image assets require a rebuild; dataset binaries are a live read-only mount. Do not change bakewell.cloud when the task is local.

## Lessons from actual checks

The directional candidate reduced acceleration slightly but worsened head and shaft reprojection; its gate retained the prior default. Same real/mirror contours caused direction conflicts and require role rejection. Native MHR was absent from older archives: rerun inference and capture it, rather than infer native parameters from exported mesh coordinates. Standard racket dimensions and inferred grip cannot establish measured handle bevels.

The native GPU refit reduced palm-anchor gaps while worsening hand/shaft direction and held-out reprojection. Gap reduction alone is insufficient for acceptance. When capturing multiple angles, use a separate review tab and verify the displayed frame after every seek; simultaneous user annotations can invalidate otherwise correctly named screenshots. Preserve invalid captures as excluded evidence rather than mixing them into the audit sheet.
