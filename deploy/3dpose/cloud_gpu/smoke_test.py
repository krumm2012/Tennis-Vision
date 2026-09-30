"""Import a real clip through the local API and wait for validated GPU output."""
import argparse,json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode,urlsplit

def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:18768');p.add_argument('--timeout',type=int,default=7200);p.add_argument('--record',type=Path);a=p.parse_args()
    base=a.base.rstrip('/')
    if urlsplit(base).hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('仅支持本地视频库')
    def call(path,data=None):
        request=Request(base+path,data=data,headers={'Content-Type':'application/octet-stream'} if data is not None else {})
        with urlopen(request,timeout=240) as response:return json.load(response)
    if not call('/api/videos')['generation_available']:raise RuntimeError('本地服务尚未配置生成命令')
    item=call('/api/videos?'+urlencode({'name':a.video.name}),a.video.read_bytes());ident=item['id']
    print('dataset_id:',ident,flush=True)
    if a.record:a.record.write_text(json.dumps(item,ensure_ascii=False,indent=2))
    call('/api/videos/'+ident+'/generate',b'')
    deadline=time.monotonic()+a.timeout;previous=None
    while time.monotonic()<deadline:
        item=next(v for v in call('/api/videos')['videos'] if v['id']==ident)
        if item['status']!=previous:print(item['status'],item['message'],flush=True);previous=item['status']
        if item['status']=='failed':raise RuntimeError(item['message'])
        if item['status']=='ready':
            report=call('/datasets/'+ident+'/result/quality_report.json')
            if report['frames']<=0:raise ValueError('空结果')
            if a.record:a.record.write_text(json.dumps({'item':item,'quality':report},ensure_ascii=False,indent=2))
            print(base+item['viewer'],flush=True);return
        time.sleep(3)
    raise TimeoutError('等待超时，任务 ID 已输出；云端任务不会自动取消')

if __name__=='__main__':main()
