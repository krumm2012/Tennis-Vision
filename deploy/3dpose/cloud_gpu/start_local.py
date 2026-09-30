"""Start the loopback video library with the SSH GPU adapter configured."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'viewer/video_import'))
from cloud_adapter import config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--port',type=int,default=18768);p.add_argument('--racket-model',type=Path);p.add_argument('--mirror-pose-model',type=Path);a=p.parse_args()
    config(a.config)
    env={**os.environ,'VIEWER_GPU_CONFIG':str(a.config.resolve()),'VIEWER_GENERATOR_COMMAND':json.dumps([sys.executable,str(ROOT/'viewer/video_import/cloud_adapter.py')])}
    if a.racket_model:env['VIEWER_RACKET_MODEL']=str(a.racket_model.resolve())
    if a.mirror_pose_model:env['VIEWER_MIRROR_MODEL']=str(a.mirror_pose_model.resolve())
    raise SystemExit(subprocess.call([sys.executable,str(ROOT/'viewer/video_import/server.py'),'--port',str(a.port)],env=env))

if __name__=='__main__':main()
