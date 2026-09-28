# bakewell.cloud /3dpose/

`prepare.py` copies the tracked Viewer and its 250-frame generated assets into ignored `output/3dpose_docker/`, excluding SAM weights and other outputs. The staged `manifest.json` records SHA-256 hashes. Build and run with:

```sh
python deploy/3dpose/prepare.py
cd output/3dpose_docker
docker compose up -d --build
```

The container binds only to `127.0.0.1:18766`. On the existing host Nginx HTTPS server, use:

```nginx
location = /3dpose { return 301 /3dpose/; }
location = /3dpose/ { return 302 /3dpose/viewer.html; }
location /3dpose/ {
    proxy_pass http://127.0.0.1:18766/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Accept-Encoding $http_accept_encoding;
    gzip on;
    gzip_types application/octet-stream application/json;
    gzip_proxied any;
}
```

The exact `/3dpose/` route opens the Viewer page. The trailing slash in `proxy_pass` removes `/3dpose/` before forwarding. The same server snippet is saved in `host-nginx-location.conf`. The default route remains with the existing service. The published page includes the user-provided video and its derived pose/mesh data; treat the URL as public.

For updates, run `prepare.py`, sync `output/3dpose_docker/` to `/home/ubuntu/tennis-3dpose/` on the host, then run `docker compose up -d --build` there. Test `nginx -t` before reloading the host Nginx if the host snippet changes.

The Viewer requests only the current frame from each 55 MB mesh file using HTTP Range. The container serves those ranges directly. It also serves precompressed pose and texture data; the host Nginx compression directives cover proxied responses.
