"""Loopback-only video library and serialized reconstruction jobs.

The trusted runner command is configured by the operator, never by HTTP input.
It receives --video PATH --output DIR and must export reconstruction.npz.
"""
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse, json, os, re, shutil, subprocess, threading, time, uuid, sys
from urllib.parse import urlsplit, parse_qs

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / 'output/sam3d_cloud'
DEFAULT_FILES = {'coaching_report.json','viewer.html','video.mp4','temporal_pose.json','mesh_meta.json','mesh_local.bin','mesh_smooth.bin','mesh_refined.bin','mesh_temporal.bin','mesh_faces.bin','temporal_texture_sam2.bin','person_masks_sam2.png','person_masks_sam2_stats.json','mirror_geometry.json','mirror_geometry_frames.json','ground_calibration.json','mirror_ground_grid.json','mirror_corner_suggestion.json','paired_ground_diagnostic.json','racket_poses.json','racket_poses_v1.json','racket_poses_v2.json','racket_poses_v3.json','racket_annotations.json','wilson_mesh.bin','wilson_model.json'}
DEFAULT_SCRIPTS = {'coaching.js','mesh_renderer.js','racket_renderer.js','texture_audit.js','calibration_editor.js','mirror_grid_editor.js','racket_editor.js','vendor/three-0.180.0.min.js'}
MAX_BYTES = 1024 * 1024 * 1024
ID = re.compile(r'^[0-9a-f]{32}$')

class Library:
    def __init__(self, root, command):
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.command = command; self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        for item in self.list():
            if item['status'] in ('queued', 'generating'):
                self.update(item['id'], status='failed', message='服务已重启，请重新生成')

    def folder(self, ident):
        if not ID.fullmatch(ident): raise ValueError('无效的视频编号')
        return self.root / ident

    def read(self, ident):
        return json.loads((self.folder(ident) / 'record.json').read_text())

    def list(self):
        with self.lock:
            return sorted([json.loads(p.read_text()) for p in self.root.glob('*/record.json')], key=lambda x:x['created'], reverse=True)

    def update(self, ident, **values):
        with self.lock:
            folder = self.folder(ident); path = folder / 'record.json'
            data = json.loads(path.read_text()) if path.exists() else {}
            data.update(values)
            temp = folder / 'record.tmp'; temp.write_text(json.dumps(data, ensure_ascii=False)); temp.replace(path)
            return data

    def generate(self, ident):
        with self.lock:
            item = self.read(ident)
            if not self.command: raise ValueError('尚未配置人体生成服务；视频已保存，可稍后重试')
            if item['status'] in ('queued','generating'): return item
            if item['status'] == 'ready': return item
            item = self.update(ident, status='queued', message='等待生成')
            self.pool.submit(self.run, ident)
            return item

    def run(self, ident):
        folder = self.folder(ident); work = folder / 'work'
        try:
            self.update(ident, status='generating', message='正在生成人体和人物遮罩')
            if work.exists(): shutil.rmtree(work)
            work.mkdir()
            with (folder / 'generation.log').open('w') as log:
                subprocess.run([*self.command,'--video',str(folder/'source.mp4'),'--output',str(work)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
            self.update(ident, message='正在校验并打包三维结果')
            from package_result import package
            package(folder/'source.mp4', work/'reconstruction.npz', work/'result')
            destination = folder/'result'
            if destination.exists(): shutil.rmtree(destination)
            (work/'result').replace(destination)
            self.update(ident, status='ready', message='生成完成', viewer=f'/datasets/{ident}/result/viewer.html')
        except Exception as exc:
            self.update(ident, status='failed', message=f'生成失败：{str(exc)[:240]}。可重试；详细日志保存在本机。')

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def allowed(self):
        # No public upload endpoint, cross-site requests, or DNS-rebinding hostnames.
        host = urlsplit('//'+self.headers.get('Host','')).hostname
        origin = self.headers.get('Origin')
        return host in ('localhost','127.0.0.1','::1') and (not origin or origin in self.server.origins)

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        origin = self.headers.get('Origin')
        if origin in self.server.origins: self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Vary','Origin'); self.send_header('Cache-Control','no-store')
        self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_OPTIONS(self):
        if not self.allowed(): return self.json({'error':'仅允许本地访问'},403)
        self.send_response(204); self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin',''))
        self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS');self.send_header('Access-Control-Allow-Headers','Content-Type')
        self.end_headers()

    def do_GET(self):
        if not self.allowed(): return self.json({'error':'仅允许本地访问'},403)
        route = urlsplit(self.path).path
        if route == '/api/videos': return self.json({'videos':self.server.library.list(),'generation_available':bool(self.server.library.command)})
        if route == '/' or route == '/import.html': path = REPO/'viewer/video_import/import.html'
        elif route.startswith('/default/'):
            name = route.removeprefix('/default/')
            if name in DEFAULT_SCRIPTS: path = REPO/'viewer/sam3d'/name
            elif name in DEFAULT_FILES:
                path = DEFAULT_DATA/'joint_fit_v4/full'/name
                if not path.is_file(): path = DEFAULT_DATA/name
            else: return self.send_error(404)
        elif route.startswith('/assets/'):

            name = route.removeprefix('/assets/')
            if name not in ('coaching.js','mesh_renderer.js','vendor/three-0.180.0.min.js'): return self.send_error(404)
            path = REPO/'viewer/sam3d'/name
        else:
            match = re.fullmatch(r'/datasets/([0-9a-f]{32})/(source.mp4|result/[a-zA-Z0-9_.-]+)',route)
            if not match: return self.send_error(404)
            path = self.server.library.folder(match[1])/match[2]
        if not path.is_file() or (path.is_symlink() and not route.startswith('/default/')): return self.send_error(404)
        # Range support for video seeking and streamed mesh frames.
        size=path.stat().st_size; start=0;end=size-1;partial=False
        if self.headers.get('Range'):
            m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
            if not m: return self.send_error(416)
            start=int(m[1]);end=min(int(m[2]) if m[2] else end,end);partial=True
            if start>end: return self.send_error(416)
        self.send_response(206 if partial else 200)
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Type',self.guess_type(str(path)));self.send_header('Content-Length',str(end-start+1))
        self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-cache');self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start);left=end-start+1
                while left:
                    chunk=stream.read(min(left,1024*1024))
                    if not chunk:break
                    self.wfile.write(chunk);left-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        if not self.allowed(): return self.json({'error':'仅允许本地访问'},403)
        route=urlsplit(self.path)
        try:
            if route.path == '/api/videos':
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BYTES: return self.json({'error':'请选择不超过 1 GB 的视频'},413)
                if shutil.disk_usage(self.server.library.root).free < max(size*3,256*1024*1024): return self.json({'error':'磁盘空间不足，请先释放空间再导入'},507)
                name=Path(parse_qs(route.query).get('name',['视频'])[0]).name[:160]
                ident=uuid.uuid4().hex;folder=self.server.library.folder(ident);folder.mkdir()
                try:
                    self.connection.settimeout(120)
                    with (folder/'upload').open('wb') as stream:
                        remaining=size
                        while remaining:
                            chunk=self.rfile.read(min(remaining,1024*1024))
                            if not chunk:raise ValueError('上传中断，请重试')
                            stream.write(chunk);remaining-=len(chunk)
                    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height:format=duration','-of','json',str(folder/'upload')],capture_output=True,text=True,check=True,timeout=30)
                    info=json.loads(probe.stdout);video=info['streams'][0];duration=float(info['format']['duration'])
                    if not 0<duration<=120:raise ValueError('当前支持最长 120 秒的视频片段')
                    # Normalize browser playback and the inference input to the same timeline.
                    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(folder/'upload'),'-map','0:v:0','-an','-vf',"scale='trunc(min(1280,iw)/2)*2':-2,fps=25",'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'source.mp4')],capture_output=True,check=True,timeout=180)
                    (folder/'upload').unlink()
                    item=self.server.library.update(ident,id=ident,name=name,created=time.time(),duration=duration,status='imported',message='导入完成，可以生成人体',preview=f'/datasets/{ident}/source.mp4')
                    return self.json(item,201)
                except Exception:
                    shutil.rmtree(folder);raise
            match=re.fullmatch(r'/api/videos/([0-9a-f]{32})/generate',route.path)
            if match:return self.json(self.server.library.generate(match[1]),202)
            return self.json({'error':'未知请求'},404)
        except (ValueError,FileNotFoundError,KeyError,IndexError,subprocess.SubprocessError, OSError) as exc:
            return self.json({'error':str(exc)[:250]},400)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18768);parser.add_argument('--root',type=Path,default=REPO/'output/video_library');args=parser.parse_args()
    command=json.loads(os.environ.get('VIEWER_GENERATOR_COMMAND','[]'))
    if not command:
        from generate_sam import requirements
        if not requirements(): command=[sys.executable,str(Path(__file__).with_name('generate_sam.py'))]
    if not isinstance(command,list) or any(not isinstance(x,str) for x in command):raise ValueError('VIEWER_GENERATOR_COMMAND must be a JSON argument array')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.library=Library(args.root,command);server.origins={f'http://{host}:{port}' for host in ('localhost','127.0.0.1') for port in (18766,args.port)}
    print(f'Video library: http://127.0.0.1:{args.port}',flush=True);server.serve_forever()

if __name__=='__main__':main()
