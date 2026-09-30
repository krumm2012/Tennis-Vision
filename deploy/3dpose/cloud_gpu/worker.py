"""Durable per-video GPU jobs over private SSH; no public HTTP listener."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
from run_records import sha256,write,now

def folder(root,job):
    if not re.fullmatch('[0-9a-f]{32}',job):raise ValueError('Invalid job ID')
    return Path(root)/'jobs'/job

def execute(root,job):
    base=folder(root,job);cfg=json.loads((Path(root)/'worker_config.json').read_text());request=json.loads((base/'request.json').read_text())
    manifest={'schema_version':1,'remote_job_id':job,'started_at':now(),'status':'generating','source_sha256':sha256(base/'source.mp4'),'git_commit':cfg['git_commit'],'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'transport':'ssh','runtime':'existing_gpu_python'}
    write(base/'status.json',{'status':'generating','pid':os.getpid()})
    try:
        if manifest['source_sha256']!=request['source_sha256']:raise ValueError('Input hash mismatch')
        env={**os.environ,**cfg['env'],'VIEWER_REQUIRE_CUDA':'1'}
        subprocess.run([sys.executable,str(Path(__file__).with_name('generate_sam.py')),'--video',str(base/'source.mp4'),'--output',str(base/'work')],env=env,check=True,timeout=6800)
        manifest['artifact_sha256']={'reconstruction.npz':sha256(base/'work/reconstruction.npz')}
        inference=base/'work/inference_manifest.json'
        if inference.exists():manifest['inference']=json.loads(inference.read_text())
        manifest['status']='ready'
    except Exception as e:
        manifest['status']='failed';manifest['error_type']=type(e).__name__
        print('Worker failed:',type(e).__name__,str(e)[:240],flush=True)
    finally:
        manifest['finished_at']=now();write(base/'run_manifest.json',manifest);write(base/'status.json',{'status':manifest['status'],'error_type':manifest.get('error_type')})
    return manifest['status']=='ready'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run','status','check']);p.add_argument('--root',type=Path,required=True);p.add_argument('--job');a=p.parse_args()
    if a.action=='check':
        cfg=json.loads((a.root/'worker_config.json').read_text());os.environ.update(cfg['env'])
        from generate_sam import requirements
        import torch
        print(json.dumps({'cuda':torch.cuda.is_available(),'missing':requirements(),'git_commit':cfg['git_commit']}));return
    base=folder(a.root,a.job)
    if a.action=='status':print((base/'status.json').read_text());return
    if a.action=='start':
        if (base/'status.json').exists():print((base/'status.json').read_text());return
        write(base/'status.json',{'status':'queued'})
        with (base/'worker.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--root',str(a.root),'--job',a.job],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'job_id':a.job,'pid':child.pid}));return
    if not execute(a.root,a.job):raise SystemExit(1)

if __name__=='__main__':main()
