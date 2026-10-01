"""Copy versioned worker sources to a configured, already-provisioned GPU host."""
import argparse,json,subprocess,sys,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'viewer/video_import'))
from cloud_adapter import config,ssh,copy
from run_records import revision,write,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=config(a.config)
    allowed={'SAM3D_BODY_CODE','SAM3D_WEIGHTS','SAM3D_FACES','VIEWER_SEGMENTATION_WEIGHTS','VIEWER_SAM2_WEIGHTS','VIEWER_MULTIVIEW'}
    env=c.get('env',{})
    if set(env)-allowed or not all(isinstance(v,str) and (v in ['0','1'] if k=='VIEWER_MULTIVIEW' else v.startswith('/')) for k,v in env.items()):raise ValueError('env 只能包含模型配置绝对路径和 VIEWER_MULTIVIEW=0/1')
    if not {'SAM3D_BODY_CODE','SAM3D_WEIGHTS','VIEWER_SEGMENTATION_WEIGHTS'}<=set(env):raise ValueError('请配置 SAM 源码、权重和分割模型路径')
    if env.get('VIEWER_MULTIVIEW')=='1' and not env.get('VIEWER_SAM2_WEIGHTS'):raise ValueError('双视角生成需要 SAM2 权重路径')
    ssh(c,['mkdir','-p',c['root']+'/code',c['root']+'/jobs'])
    hashes={}
    for source in [Path(__file__).with_name('worker.py'),ROOT/'viewer/video_import/generate_sam.py',ROOT/'viewer/video_import/run_records.py',ROOT/'viewer/video_import/generate_multiview.py',ROOT/'viewer/video_import/mhr_parameters.py',ROOT/'viewer/video_import/refit_fullbody.py',ROOT/'viewer/video_import/grip_objective.py',ROOT/'viewer/video_import/refit_observations.py',ROOT/'viewer/video_import/refit_readiness.py',ROOT/'viewer/video_import/racket_calibration.py',ROOT/'viewer/video_import/racket_landmarks.py',ROOT/'viewer/video_import/export_mhr_uv.py']:
        hashes[source.name]=sha256(source)
        copy(c,source,c['root']+'/code/'+source.name,upload=True)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'worker_config.json';write(path,{'git_commit':revision(ROOT),'env':env,'code_sha256':hashes});copy(c,path,c['root']+'/worker_config.json',upload=True)
    check=json.loads(ssh(c,[c['python'],c['root']+'/code/worker.py','check','--root',c['root']]))
    print(json.dumps(check,ensure_ascii=False))
    if not check.get('cuda') or check.get('missing'):raise RuntimeError('GPU 环境检查未通过')

if __name__=='__main__':main()
