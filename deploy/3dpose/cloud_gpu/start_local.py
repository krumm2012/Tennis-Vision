"""Start the loopback video library with the SSH GPU adapter configured."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'viewer/video_import'))
from cloud_adapter import config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--port',type=int,default=18768);a=p.parse_args()
    config(a.config)
    env={**os.environ,'VIEWER_GPU_CONFIG':str(a.config.resolve()),'VIEWER_GENERATOR_COMMAND':json.dumps([sys.executable,str(ROOT/'viewer/video_import/cloud_adapter.py')])}
    raise SystemExit(subprocess.call([sys.executable,str(ROOT/'viewer/video_import/server.py'),'--port',str(a.port)],env=env))

if __name__=='__main__':main()
