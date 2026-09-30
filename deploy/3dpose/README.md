# bakewell.cloud /3dpose/

`prepare.py` copies the tracked Viewer and its 250-frame generated assets into ignored `output/3dpose_docker/`, excluding SAM weights and other outputs. The staged `manifest.json` records SHA-256 hashes. Build and run with:

```sh
python deploy/3dpose/prepare.py
cd output/3dpose_docker
docker compose up -d --build
```

The container binds only to `127.0.0.1:18766`. On the existing host Nginx,
the upstream belongs inside the `http` block:

```nginx
upstream tennis_3dpose_viewer {
    server 127.0.0.1:18767 max_fails=1 fail_timeout=5s;
    server 127.0.0.1:18766 backup;
}
```

The HTTPS `server` block uses:

```nginx
location = /3dpose { return 301 /3dpose/; }
location = /3dpose/ { return 302 /3dpose/viewer.html; }
location /3dpose/ {
    proxy_pass http://tennis_3dpose_viewer/;
    proxy_next_upstream error timeout http_502 http_503 http_504;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Accept-Encoding $http_accept_encoding;
    gzip on;
    gzip_types application/octet-stream application/json;
    gzip_proxied any;
}
```

The exact `/3dpose/` route opens the Viewer page. The trailing slash in `proxy_pass` removes `/3dpose/` before forwarding. The configuration snippets are saved in `host-nginx-upstream.conf` and `host-nginx-location.conf`. The default route remains with the existing service. The published page includes the user-provided video and its derived pose/mesh data; treat the URL as public.

For updates, run `prepare.py`, copy the generated release into a new directory under `/home/ubuntu/tennis-3dpose-releases/`, build a uniquely tagged image, verify it on a temporary loopback port, then switch the `tennis-3dpose` Compose project to the new release. Keep the previous release and image for rollback. Test `nginx -t` before reloading the host Nginx if the host snippet changes.

The Viewer requests only the current frame from each 55 MB mesh file using HTTP Range. The container serves those ranges directly. It also serves precompressed pose and texture data; the host Nginx compression directives cover proxied responses.


## Local Docker release (2026-09-29)

The local deployment includes baseline Viewer, Wilson mesh, v1/v2/v3 comparisons,
sequence diagnostic Viewer, and the complete native-body/joint-racket candidate.
`prepare.py` dereferences all candidate asset links and writes a 60-file SHA256
manifest to both the build folder and `public/release_manifest.json`. Model
weights, SSH credentials and raw inference archives are not packaged.

Local URLs:

- Baseline: http://127.0.0.1:18766/viewer.html
- Full candidate: http://127.0.0.1:18766/joint_fit_v4/full/viewer.html
- Joint diagnostic: http://127.0.0.1:18766/joint_fit_v4/viewer.html

Docker Desktop was started and the existing `tennis-3dpose-viewer` container was
rebuilt. Docker Hub metadata lookup for the old pinned base stalled; this local
release uses the already cached `nginx:stable-alpine` image via the configurable
`VIEWER_BASE_IMAGE` build argument. The local build folder `.env` preserves that
choice. This section records the local build; the remote publication is below.

```sh
python3 deploy/3dpose/prepare.py
VIEWER_BASE_IMAGE=nginx:stable-alpine docker compose -f output/3dpose_docker/compose.yaml up -d --build
```

HTML, scripts, JSON and binary resources now revalidate with `Cache-Control:
no-cache`, so same-path dataset updates do not intentionally remain cached for
an hour. HTTP Range and precompressed pose/texture data remain supported.

Validation: all 60 deployed file hashes and HTTP resources; byte-range responses
for video, mesh and Wilson model; gzip responses; Nginx configuration and health
endpoint. The initial release displayed orange joints as an overlay without
deforming the hand mesh. Later local builds may include an approximate hand mesh
preview; check the candidate page and release manifest for the running version.
This release is bound to local loopback.

## bakewell.cloud release (2026-09-29)

Release `20260929-joint-v4` is stored on the host at
`/home/ubuntu/tennis-3dpose-releases/20260929-joint-v4/`. It runs as
`tennis-3dpose-viewer:release-20260929-joint-v4` in the existing
`tennis-3dpose` Compose project. The host uses its cached
`nginx:1.27-alpine` base image. The existing host HTTPS reverse proxy serves:

- Baseline: https://bakewell.cloud/3dpose/viewer.html
- Complete candidate: https://bakewell.cloud/3dpose/joint_fit_v4/full/viewer.html
- Joint diagnostic: https://bakewell.cloud/3dpose/joint_fit_v4/viewer.html

All 60 staged files matched the local SHA-256 manifest before promotion. The
temporary container passed Nginx, health, HTTP asset, gzip and byte-range
checks. After promotion, the production container was healthy, and the six key
public HTML/JSON resources matched the release manifest byte-for-byte. The
public mesh and Wilson model returned valid 206 byte-range responses.

For rollback, the former release directory and image tag remain available:

```sh
ssh ubuntu@124.222.243.71
cd /home/ubuntu/tennis-3dpose
docker compose -p tennis-3dpose up -d --no-build --pull never
```

The release was built from the local working tree at Git commit
`bcbc0efa78be2898d7524cf90504ce07e5c3eb0b` with uncommitted changes. The
commit alone does not reproduce this release; use the staged release directory
and its manifest. This is a static Viewer and needs no cloud GPU at runtime.

## Local Docker and bakewell.cloud (updated 2026-09-30)

The Mac runs the local Docker Viewer on `127.0.0.1:18766`. A macOS LaunchAgent
keeps an SSH reverse tunnel open from the host's loopback port `18767` to that
local port. Host Nginx sends `/3dpose/` to the host container on `18766` first,
using the tunnel as a backup. Mesh frames are too large to stream through the
tunnel smoothly: an eight-frame public range took 2.6 seconds through the
tunnel, compared with 0.32 seconds of video at 25 fps. The Viewer now pauses
and preserves the last textured mesh while waiting for a missing frame, then
resumes automatically. No host port is exposed publicly for the tunnel.

The checked-in LaunchAgent source is `com.hehaa.tennis-3dpose-tunnel.plist`.
It is installed at
`~/Library/LaunchAgents/com.hehaa.tennis-3dpose-tunnel.plist` and loaded in the
current user's GUI session. It uses the existing SSH key, checks the connection
every 20 seconds, and restarts after disconnection. The Mac must be powered on,
awake, logged in, and running Docker for the local backup to be available.
After editing assets, rebuild the local Docker image for local testing. To
update the public page, package and deploy a new host release; local Docker
changes no longer appear on the public page automatically. If the host
container is unavailable, the tunnel serves the local version while the Mac
and Docker are running.

Validation: the tunnel returned both Viewer pages; public HTML, candidate JSON,
mesh and Wilson byte ranges succeeded after Nginx switched to the tunnel.
Temporarily unloading the LaunchAgent yielded a public 200 response from the
host fallback, then the tunnel was loaded again. The host Nginx configuration
passed `nginx -t`. The prior configuration is saved on the host at
`/etc/nginx/nginx.conf.bak-3dpose-local-20260929`.

The current host release is
`/home/ubuntu/tennis-3dpose-releases/20260930-threejs/`, image
`tennis-3dpose-viewer:release-20260930-threejs`. The previous Nginx
configuration is saved at `/etc/nginx/nginx.conf.bak-3dpose-direct-20260930`;
older release directories and image tags remain available for rollback.

This release replaces direct WebGL rendering with Three.js 0.180.0 for the
body, racket, hand joints, ground, trajectory and skeleton. The library and
license are packaged at both routes; no external CDN is required. Both local
250-frame texture audits completed with matching frame IDs and visible mesh
coverage on every frame, including mirror and temporal texture paths. Camera
regression checks cover 96 view/point combinations. The previous
`20260930-buffered-v2` release remains the rollback target.

## Independent local Docker for imported videos

Use a separate `tennis-3dpose-local` container on **127.0.0.1:18769**. This does not replace the 18766 container used by the public tunnel. The image contains the video-library UI and current renderer/tool scripts; a read-only mount serves ready datasets. Only source video and files under result/ are exposed, not attempt logs or NPZ working directories. The companion Mac API remains on 18768 for marker writes, MPS postprocessing and private cloud-GPU dispatch.

```sh
# First start the Mac companion service (in its own terminal).
python3 deploy/3dpose/cloud_gpu/start_local.py \
  --config deploy/3dpose/cloud_gpu/host.local.json \
  --racket-model /absolute/path/yolo26s-seg.pt \
  --mirror-pose-model /absolute/path/yolo26m-pose.pt

python3 deploy/3dpose/prepare_local.py
VIEWER_BASE_IMAGE=nginx:stable-alpine docker compose \
  -p tennis-3dpose-local -f output/3dpose_local/compose.yaml \
  --env-file output/3dpose_local/.env up -d --build
```

Entry: http://127.0.0.1:18769/import.html
Current clip: http://127.0.0.1:18769/datasets/85ade7a072984579831f5cb76e8e5fd3/result/viewer.html
Default comparison: http://127.0.0.1:18769/default/viewer.html

`prepare_local.py` records UI hashes and the Git revision in release_manifest.json, and resolves the video-library mount into an ignored .env file. Rebuild after UI code changes; mounted generated data and marker changes are visible without rebuilding. Docker Desktop must resolve host.docker.internal to the Mac service. Nginx forwards a fixed loopback Host, while the API checks Origin including the local Docker port. No SSH keys or model weights are built into the image.

Verification: nginx -t, /healthz, video-library API with Origin 18769, byte Range for video/mesh, body/racket playback and mirror on/off comparison. Current cached base image digest: nginx@sha256:985220252f3863977e468f611ef118ebd01421289dd86ee1ae99cb068c3bce2b. Public deployment config and tunnel are separate.
