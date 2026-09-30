# Local video import and generation

Default Viewer via local library: http://127.0.0.1:18768/default/viewer.html

The library serves the existing default dataset directly (including HTTP Range),
so its default-video link does not depend on the Docker service on port 18766.
Library: http://127.0.0.1:18768/import.html

Start the companion service on the Mac (nginx stays a static Docker deployment):

```sh
python3 viewer/video_import/server.py
```

Uploads are stored under ignored `output/video_library/<random id>/`. Imported
videos are normalized to 25 fps, max width 1280, H.264 and no audio. The original
upload is removed after conversion. Limits: 1 GB and 120 seconds. The current
sample dataset is never changed or automatically replaced. Video library access
is bound to loopback and checks Host and Origin. It is not a public upload API.

## Real generation prerequisites

The built-in runner requires an already installed SAM 3D Body environment and
locally provisioned model assets. Set:

- `SAM3D_BODY_CODE`: SAM source directory
- `SAM3D_WEIGHTS`: directory with model.ckpt and assets/mhr_model.pt
- `SAM3D_FACES`: .npy triangle topology matching those SAM vertices
- `VIEWER_SEGMENTATION_WEIGHTS`: local Ultralytics person segmentation weights

Use the Python environment containing torch, opencv, numpy, ultralytics and SAM
dependencies to start the service. It automatically enables the built-in runner
when these paths exist. No model weights are downloaded automatically. Without
these prerequisites upload/preview remains available and generation is disabled.
Restart the service after configuring models. Imported videos survive restart.

The built-in runner chooses the largest initial person and follows box overlap;
it fails explicitly on loss of tracking. Multi-person target selection, SAM2
tracking, racket reconstruction, per-video court/mirror calibration and temporal
texture fusion remain separate work. No copied old calibration is used.

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
published only after packaging succeeds; jobs are serialized, have a two-hour
timeout, and interrupted jobs become retryable failures on restart. Errors are
shown in the library; detailed runner logs stay in each local dataset folder.
Do not reuse the default video's output as a generator for a new video.

```sh
python3 -m unittest discover -s viewer/video_import -p 'test_*.py' -v
```

Tests use generated color videos and synthetic geometry, not model inference.
