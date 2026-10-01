"""Upload scoped observations and dispatch a durable private GPU refit job.

Requires a verified native-output folder from cloud_adapter. Never publishes
returned MHR meshes to the current viewer.
"""
import argparse
import json
import time
import uuid
import subprocess
import warnings
from pathlib import Path
from cloud_adapter import config,ssh,copy
from refit_readiness import assess
from run_records import write,sha256
from mhr_fit_progress import record_fit,DEFAULT_LEDGER


def run(dataset,native_output,output,c,allow_assumed=False,allow_automatic=False,progress_ledger=DEFAULT_LEDGER):
    """Record every attempt and outcome while preserving the runner's return API."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    context={'run_id':uuid.uuid4().hex,'current_files':['progress_attempt.json'],'phase':'preparing'}
    write(out/'progress_attempt.json',{'run_id':context['run_id'],'output':str(out.resolve()),'remote_job_id':None})
    def checkpoint(phase,state=None):
        context['phase']=phase
        try:
            return record_fit(out,progress_ledger,context=context,execution_state=state,phase=phase)
        except Exception as error:
            # Recording is a side effect. A disk/ledger error must not replace
            # the original GPU/SSH failure or alter the runner's return value.
            context.setdefault('progress_recording_errors',[]).append({'phase':phase,'type':type(error).__name__})
            warnings.warn(f'MHR progress record failed at {phase}: {type(error).__name__}; retain output for backfill.',RuntimeWarning)
    checkpoint('preparing','pending')
    try:
        value=_execute(dataset,native_output,out,c,allow_assumed,allow_automatic,context,checkpoint)
    except Exception as error:
        transport=isinstance(error,(OSError,subprocess.SubprocessError,TimeoutError))
        context['error']={'type':type(error).__name__,'message':str(error)[:400] if isinstance(error,(ValueError,TimeoutError)) else 'See retained local/worker logs; transport or execution error.'}
        context['verification_failed']=isinstance(error,ValueError) and '哈希' in str(error)
        # An acknowledged or attempted dispatch may continue remotely when SSH
        # disconnects. Do not label the remote computation as failed in that case.
        state='transport_pending' if transport and context.get('dispatch_attempted') and context.get('remote_status') not in ['failed','needs_input'] else 'failed'
        checkpoint('exception',state)
        raise
    checkpoint('finished')
    return value


def _execute(dataset,native_output,output,c,allow_assumed,allow_automatic,context,checkpoint):
    folder=Path(dataset);native=Path(native_output);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    readiness=assess(folder/'result',native/'reconstruction.npz',allow_assumed,allow_automatic);write(out/'readiness.json',readiness)
    context['current_files'].append('readiness.json');context['video_sha256']=readiness.get('video_sha256');context['archive_sha256']=readiness.get('archive_sha256')
    checkpoint('readiness','pending' if readiness['ready'] else 'needs_input')
    if not readiness['ready']:return readiness
    parent=json.loads((native/'remote_job.json').read_text())['job_id']
    if len(parent)!=32 or any(k not in '0123456789abcdef' for k in parent):raise ValueError('无效的来源 GPU job')
    prior=json.loads((native/'remote_manifest.json').read_text());archive_sha=sha256(native/'reconstruction.npz')
    if prior['status']!='ready' or prior['source_sha256']!=readiness['video_sha256'] or prior['artifact_sha256']['reconstruction.npz']!=archive_sha:raise ValueError('原生 GPU 结果来源不匹配')
    job=context['run_id'];remote=f"{c['root']}/jobs/{job}";worker=f"{c['root']}/code/worker.py"
    write(out/'remote_job.json',{'job_id':job,'parent_job':parent,'kind':'fullbody_refit','video_sha256':readiness['video_sha256']})
    write(out/'progress_attempt.json',{'run_id':context['run_id'],'output':str(out.resolve()),'remote_job_id':job})
    context.update(remote_job_id=job,parent_job_id=parent);context['current_files'].append('remote_job.json');checkpoint('uploading','pending')
    ssh(c,['mkdir','-p',remote+'/refit_input']);ssh(c,['ln',f"{c['root']}/jobs/{parent}/source.mp4",remote+'/source.mp4'])
    for name in ['mesh_meta.json','racket_poses.json','racket_poses_directional.json','racket_keypoints.json','racket_review_frames.json','racket_dimensions.json','racket_landmarks.json','mirror_geometry.json']:
        source=folder/'result'/name
        if source.exists():copy(c,source,remote+'/refit_input/'+name,upload=True)
    request={'kind':'fullbody_refit','base_job_id':parent,'source_sha256':readiness['video_sha256'],'archive_sha256':archive_sha,'allow_assumed':allow_assumed,'allow_automatic':allow_automatic}
    write(out/'request.json',request);copy(c,out/'request.json',remote+'/request.json',upload=True)
    context['current_files'].append('request.json');context['dispatch_attempted']=True;checkpoint('dispatching','pending')
    ssh(c,[c['python'],worker,'start','--root',c['root'],'--job',job]);context['dispatch_acknowledged']=True;checkpoint('dispatched','pending');deadline=time.monotonic()+6900;last=None
    try:
        while time.monotonic()<deadline:
            for retry in range(4):
                try:
                    status=json.loads(ssh(c,[c['python'],worker,'status','--root',c['root'],'--job',job]))['status'];break
                except (OSError,subprocess.SubprocessError):
                    if retry==3:raise
                    time.sleep(3*(retry+1))
            context['remote_status']=status
            if status!=last:print('GPU fullbody refit',job,status,flush=True);last=status;checkpoint('remote_'+status,'pending' if status in ['queued','generating'] else None)
            if status in ['ready','failed','needs_input']:break
            time.sleep(2)
        else:raise TimeoutError('重拟合等待超时，远端任务仍保留')
        copy(c,remote+'/run_manifest.json',out/'remote_manifest.json')
        context['current_files'].append('remote_manifest.json')
        manifest=json.loads((out/'remote_manifest.json').read_text())
        if status!='ready':return manifest
        copy(c,remote+'/refit_output/mhr_refit_candidate.npz',out/'mhr_refit_candidate.npz.part')
        if manifest['source_sha256']!=readiness['video_sha256'] or sha256(out/'mhr_refit_candidate.npz.part')!=manifest['artifact_sha256']['mhr_refit_candidate.npz']:raise ValueError('重拟合候选哈希不匹配')
        (out/'mhr_refit_candidate.npz.part').replace(out/'mhr_refit_candidate.npz')
        context['current_files'].append('mhr_refit_candidate.npz')
        copy(c,remote+'/refit_output/refit_report.json',out/'refit_report.json');context['current_files'].append('refit_report.json');return manifest
    finally:
        try:copy(c,remote+'/worker.log',out/'worker.log');context['current_files'].append('worker.log')
        except Exception:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--native-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config')
    p.add_argument('--allow-assumed',action='store_true');p.add_argument('--allow-automatic',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.dataset,a.native_output,a.output,config(a.config),a.allow_assumed,a.allow_automatic),ensure_ascii=False))
