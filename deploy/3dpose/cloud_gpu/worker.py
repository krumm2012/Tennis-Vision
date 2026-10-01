"""Durable per-video GPU jobs over private SSH; no public HTTP listener."""
import argparse,json,os,re,subprocess,sys,time,shutil
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
        if request.get('kind')=='fullbody_refit':
            previous=folder(root,request['base_job_id']);prior=json.loads((previous/'run_manifest.json').read_text());archive=previous/'work/reconstruction.npz'
            if prior['status']!='ready' or prior['source_sha256']!=manifest['source_sha256'] or sha256(archive)!=request['archive_sha256']:raise ValueError('Refit parent identity/hash mismatch')
            args=[sys.executable,str(Path(__file__).with_name('refit_fullbody.py')),'--result',str(base/'refit_input'),'--archive',str(archive),'--output',str(base/'refit_output')]
            if request.get('allow_assumed'):args.append('--allow-assumed')
            if request.get('allow_automatic'):args.append('--allow-automatic')
            subprocess.run(args,env=env,check=True,timeout=6800)
            readiness=json.loads((base/'refit_output/readiness.json').read_text());manifest.update(kind='fullbody_refit',readiness=readiness)
            if not readiness['ready']:
                manifest['status']='needs_input';return False
            report=json.loads((base/'refit_output/refit_report.json').read_text());manifest.update(refit=report,publication_status='needs_review',artifact_sha256={'mhr_refit_candidate.npz':sha256(base/'refit_output/mhr_refit_candidate.npz')},status='ready')
            return True
        parent=request.get('base_job_id')
        if parent:
            previous=folder(root,parent);prior=json.loads((previous/'run_manifest.json').read_text())
            if prior['status']!='ready' or prior['source_sha256']!=manifest['source_sha256']:raise ValueError('Reuse requires same video hash and a ready cloud job')
            if sha256(previous/'work/reconstruction.npz')!=prior['artifact_sha256']['reconstruction.npz']:raise ValueError('Parent artifact hash mismatch')
            import numpy as np
            from mhr_parameters import available
            with np.load(previous/'work/reconstruction.npz',allow_pickle=False) as archive:native=available(archive,len(archive['vertices']))
            if native:
                (base/'work').mkdir(exist_ok=True);shutil.copy2(previous/'work/reconstruction.npz',base/'work/reconstruction.npz');write(base/'work/inference_manifest.json',{**prior['inference'],'primary_reused_from_job':parent,'reuse_source_sha256':manifest['source_sha256']})
            else:
                subprocess.run([sys.executable,str(Path(__file__).with_name('generate_sam.py')),'--video',str(base/'source.mp4'),'--output',str(base/'work')],env=env,check=True,timeout=6800)
                manifest['primary_reinferred_for_native_mhr']=parent
        else:
            subprocess.run([sys.executable,str(Path(__file__).with_name('generate_sam.py')),'--video',str(base/'source.mp4'),'--output',str(base/'work')],env=env,check=True,timeout=6800)
        if env.get('VIEWER_MULTIVIEW')=='1':
            subprocess.run([sys.executable,str(Path(__file__).with_name('generate_multiview.py')),'--video',str(base/'source.mp4'),'--output',str(base/'work')],env=env,check=True,timeout=6800)
            manifest['multiview']=json.loads((base/'work/multiview_manifest.json').read_text())
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
        missing=requirements()
        if cfg['env'].get('VIEWER_MULTIVIEW')=='1':
            import importlib.util
            if importlib.util.find_spec('sam2') is None:missing.append('sam2 python package')
            if not Path(cfg['env'].get('VIEWER_SAM2_WEIGHTS','')).is_file():missing.append('VIEWER_SAM2_WEIGHTS')
        print(json.dumps({'cuda':torch.cuda.is_available(),'missing':missing,'git_commit':cfg['git_commit']}));return
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
