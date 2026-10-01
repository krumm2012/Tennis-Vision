---
name: tennis-racket-fitting
description: Fit and audit tennis racket motion and hand grip using hand keypoints, racket observations and optional mirror views. Use for racket orientation conflicts, jitter, missing frames, grip contact or multi-angle fit review.
---

# Tennis racket fitting

Produce a reproducible candidate and a decision supported by source-video alignment, grip contact and motion evidence. Read [the project workflow](references/tennis-vision.md) when working with Tennis-Vision.

## Evidence and coordinates

- Bind every observation, camera, mirror plane and output to the normalized video's SHA-256, dimensions, fps and frame indices. Preserve originals and accepted results.
- Distinguish actual and reflected people. Associate rackets using their respective wrist and tracked person boxes. Reject a contour assigned to both roles. A reflection is a virtual camera; restore its handedness exactly once. SAM MHR anatomical indices and COCO mirror labels require different correspondence handling.
- Retain high-resolution hand/racket crops and map observations back to full-image pixels. Use resolution-normalized residual thresholds; do not upsample low-resolution evidence and call it 4K analysis.
- A symmetric silhouette constrains an unoriented plane. Continuity does not establish physical face A/B or handle bevel. Use visible asymmetric markings or reviewed correspondence for those claims.

## Fit

Use MCP/PIP finger corridors and wrist geometry to estimate the grip center and shaft direction, following the demo's anatomical method. Weight unreliable hand direction down when clear racket observations conflict. Add butt, throat, tip and rim landmarks, plus mirror evidence only after held-out plane validation. Label asset dimensions as assumptions until measured.

Fit a rigid racket with robust reprojection losses and SO(3) temporal constraints. Complete only short, bounded gaps supported at both ends; expose unsupported frames. Preserve fast swings when regularizing acceleration. Keep alternate candidates separate from accepted geometry and motion.

For body refitting, retain native MHR pose/shape/expression parameters during inference. Verify native replay against source vertices and joints before changing pose. A display-space hand correction or a finger-joint cylinder proxy is not anatomical mesh contact or a full-body fit.

## Review and acceptance

Compare the same timeline and fixed observations across candidates. Inspect front, back, left/right and oblique views, with a grip close-up, at distributed clear frames and throughout playback. Also inspect source-video and mirror reprojection: attractive novel views cannot prove accuracy.

Report coverage/hidden frames, angular steps and acceleration, head/shaft reprojection, contact gaps, and held-out errors separately. Reject candidates that smooth motion while worsening observation alignment or anatomy. Preserve source hashes, code/model versions, selected/held-out frames, assumptions, numerical reports and screenshots. Publish only after the relevant gates and visual review pass; automatic clear-frame selection remains unreviewed evidence.
