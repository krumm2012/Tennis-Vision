"""Private SSH transport for the existing --video / --output runner contract."""
import argparse,json,os,re,shlex,subprocess,time,uuid
from pathlib import Path
from run_records import sha256,write

SSH_OPTIONS=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes']

def config(path=None):
    path=path or os.environ.get('VIEWER_GPU_CONFIG')
    if not path:raise ValueError('请配置 VIEWER_GPU_CONFIG，指向 GPU 主机配置 JSON')
    c=json.loads(Path(path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',c.get('host','')):raise ValueError('无效的 SSH host/别名')
    for k in ('root','python'):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+',c.get(k,'')) or '..' in c[k].split('/'):raise ValueError(f'{k} 必须是无空格的绝对路径')
    port=c.get('port',22)
    if not isinstance(port,int) or not 1<=port<=65535:raise ValueError('无效的 SSH port')
    return c

def ssh(c,args):
    return subprocess.run(['ssh',*SSH_OPTIONS,'-p',str(c.get('port',22)),c['host'],shlex.join([str(a) for a in args])],capture_output=True,text=True,check=True,timeout=90).stdout

def copy(c,source,dest,upload=False):
    remote=f"{c['host']}:{source if not upload else dest}"
    args=[str(source),remote] if upload else [remote,str(dest)]
    subprocess.run(['scp',*SSH_OPTIONS,'-P',str(c.get('port',22)),*args],check=True,timeout=600)

def run(video,output,c,poll=2,timeout=13800,base_job=None):
    if base_job and not re.fullmatch('[0-9a-f]{32}',base_job):raise ValueError('无效 parent GPU job')
    video=Path(video).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    job=uuid.uuid4().hex;folder=f"{c['root']}/jobs/{job}";worker=f"{c['root']}/code/worker.py"
    write(output/'remote_job.json',{'job_id':job,'source_sha256':sha256(video),'transport':'ssh'})
    ssh(c,['mkdir','-p',folder]);copy(c,video,folder+'/source.mp4',upload=True)
    request={'job_id':job,'source_sha256':sha256(video)}
    if base_job:
        request['base_job_id']=base_job
    write(output/'request.json',request)
    copy(c,output/'request.json',folder+'/request.json',upload=True)
    ssh(c,[c['python'],worker,'start','--root',c['root'],'--job',job])
    return collect(video,output,c,job,poll,timeout)

def collect(video,output,c,job,poll=2,timeout=13800):
    """Resume an existing durable job; never dispatch duplicate inference."""
    if not re.fullmatch('[0-9a-f]{32}',job):raise ValueError('无效 GPU job')
    video=Path(video);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    request={'source_sha256':sha256(video)}
    folder=f"{c['root']}/jobs/{job}";worker=f"{c['root']}/code/worker.py"
    deadline=time.monotonic()+timeout
    last_status=None;last_message=0
    try:
        while time.monotonic()<deadline:
            state=json.loads(ssh(c,[c['python'],worker,'status','--root',c['root'],'--job',job]))
            if state['status']!=last_status or time.monotonic()-last_message>=30:
                print('GPU task',job,state['status'],flush=True)
                last_status=state['status'];last_message=time.monotonic()
            if state['status'] in ('ready','failed'):break
            time.sleep(poll)
        else:raise TimeoutError('云任务等待超时；远端任务和日志已保留')
        if state['status']=='failed':raise RuntimeError('GPU 推理失败；请检查返回的 worker.log')
        copy(c,folder+'/work/reconstruction.npz',output/'reconstruction.npz.part')
        copy(c,folder+'/run_manifest.json',output/'remote_manifest.json')
        manifest=json.loads((output/'remote_manifest.json').read_text())
        if manifest.get('source_sha256')!=request['source_sha256']:raise ValueError('云结果与当前视频不匹配')
        if sha256(output/'reconstruction.npz.part')!=manifest['artifact_sha256']['reconstruction.npz']:raise ValueError('云结果哈希校验失败')
        (output/'reconstruction.npz.part').replace(output/'reconstruction.npz')
        if manifest.get('multiview'):write(output/'multiview_manifest.json',manifest['multiview'])
    finally:
        for name,target in [('worker.log','worker.log'),('run_manifest.json','remote_manifest.json')]:
            try:copy(c,folder+'/'+name,output/target)
            except (subprocess.SubprocessError,OSError):pass
    return job

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config');p.add_argument('--base-job');a=p.parse_args()
    run(a.video,a.output,config(a.config),base_job=a.base_job)
