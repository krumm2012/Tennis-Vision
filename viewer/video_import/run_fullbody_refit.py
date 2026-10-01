"""Upload scoped observations and dispatch a durable private GPU refit job.

Requires a verified native-output folder from cloud_adapter. Never publishes
returned MHR meshes to the current viewer.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from cloud_adapter import config,ssh,copy
from refit_readiness import assess
from run_records import write,sha256


def run(dataset,native_output,output,c,allow_assumed=False,allow_automatic=False):
    folder=Path(dataset);native=Path(native_output);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    readiness=assess(folder/'result',native/'reconstruction.npz',allow_assumed,allow_automatic);write(out/'readiness.json',readiness)
    if not readiness['ready']:return readiness
    parent=json.loads((native/'remote_job.json').read_text())['job_id']
    if len(parent)!=32 or any(k not in '0123456789abcdef' for k in parent):raise ValueError('无效的来源 GPU job')
    prior=json.loads((native/'remote_manifest.json').read_text());archive_sha=sha256(native/'reconstruction.npz')
    if prior['status']!='ready' or prior['source_sha256']!=readiness['video_sha256'] or prior['artifact_sha256']['reconstruction.npz']!=archive_sha:raise ValueError('原生 GPU 结果来源不匹配')
    job=uuid.uuid4().hex;remote=f"{c['root']}/jobs/{job}";worker=f"{c['root']}/code/worker.py"
    write(out/'remote_job.json',{'job_id':job,'parent_job':parent,'kind':'fullbody_refit','video_sha256':readiness['video_sha256']})
    ssh(c,['mkdir','-p',remote+'/refit_input']);ssh(c,['ln',f"{c['root']}/jobs/{parent}/source.mp4",remote+'/source.mp4'])
    for name in ['mesh_meta.json','racket_poses.json','racket_poses_directional.json','racket_keypoints.json','racket_review_frames.json','racket_dimensions.json','racket_landmarks.json','mirror_geometry.json']:
        source=folder/'result'/name
        if source.exists():copy(c,source,remote+'/refit_input/'+name,upload=True)
    request={'kind':'fullbody_refit','base_job_id':parent,'source_sha256':readiness['video_sha256'],'archive_sha256':archive_sha,'allow_assumed':allow_assumed,'allow_automatic':allow_automatic}
    write(out/'request.json',request);copy(c,out/'request.json',remote+'/request.json',upload=True)
    ssh(c,[c['python'],worker,'start','--root',c['root'],'--job',job]);deadline=time.monotonic()+6900;last=None
    try:
        while time.monotonic()<deadline:
            for retry in range(4):
                try:
                    status=json.loads(ssh(c,[c['python'],worker,'status','--root',c['root'],'--job',job]))['status'];break
                except (OSError,__import__('subprocess').SubprocessError):
                    if retry==3:raise
                    time.sleep(3*(retry+1))
            if status!=last:print('GPU fullbody refit',job,status,flush=True);last=status
            if status in ['ready','failed','needs_input']:break
            time.sleep(2)
        else:raise TimeoutError('重拟合等待超时，远端任务仍保留')
        copy(c,remote+'/run_manifest.json',out/'remote_manifest.json')
        manifest=json.loads((out/'remote_manifest.json').read_text())
        if status!='ready':return manifest
        copy(c,remote+'/refit_output/mhr_refit_candidate.npz',out/'mhr_refit_candidate.npz.part')
        if manifest['source_sha256']!=readiness['video_sha256'] or sha256(out/'mhr_refit_candidate.npz.part')!=manifest['artifact_sha256']['mhr_refit_candidate.npz']:raise ValueError('重拟合候选哈希不匹配')
        (out/'mhr_refit_candidate.npz.part').replace(out/'mhr_refit_candidate.npz')
        copy(c,remote+'/refit_output/refit_report.json',out/'refit_report.json');return manifest
    finally:
        try:copy(c,remote+'/worker.log',out/'worker.log')
        except Exception:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--native-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config')
    p.add_argument('--allow-assumed',action='store_true');p.add_argument('--allow-automatic',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.dataset,a.native_output,a.output,config(a.config),a.allow_assumed,a.allow_automatic),ensure_ascii=False))
