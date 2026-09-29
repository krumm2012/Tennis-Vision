ARG VIEWER_BASE_IMAGE=nginx:1.27-alpine
FROM ${VIEWER_BASE_IMAGE}
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY public/ /usr/share/nginx/html/
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
