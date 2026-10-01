"""Resumeable sequential GPU generation and local per-clip mirror packaging.

Each video has private originals, a normalized timeline, verified raw/native MHR,
SAM2 evidence and its own calibration. Never overwrites the accepted Viewer.
"""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
import cv2
from cloud_adapter import config,run,collect
from run_records import sha256,write,now
from package_result import package

ROOT=Path(__file__).resolve().parents[2]

def prepare(source,folder,reuse_video=None):
    folder.mkdir(parents=True,exist_ok=True)
    identity=sha256(source);record=folder/'record.json'
    if record.exists() and json.loads(record.read_text())['original_sha256']!=identity:raise ValueError('批次源文件已改变')
    if not (folder/'source.mp4').exists():
        if reuse_video:shutil.copy2(reuse_video,folder/'source.mp4')
        else:
            temp=folder/'source.pending.mp4'
            subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(source),'-map','0:v:0','-an','-vf',"scale='trunc(min(2560,iw)/2)*2':-2,fps=25",'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True,timeout=180)
            temp.replace(folder/'source.mp4')
    if not (folder/'original.video').exists():shutil.copy2(source,folder/'original.video')
    if not record.exists():write(record,{'id':folder.name,'original_name':source.name,'original_sha256':identity,'source_sha256':sha256(folder/'source.mp4'),'attempt':1,'status':'prepared','created':now()})

def evidence(folder):
    out=folder/'result';meta=json.loads((out/'mesh_meta.json').read_text());report=json.loads((out/'multiview_manifest.json').read_text())
    frames=[]
    for row in report['tracking']:
        persons=[]
        for role in ['real','mirror']:
            box=row[role+'_box']
            if box is not None:
                x0,y0,x1,y1=box
                persons.append({'box':box,'confidence':row[role+'_confidence'],'polygon':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'role':role})
        frames.append({'frame':row['frame'],'persons':persons})
    write(out/'person_candidates.json',{'video_sha256':meta['video_sha256'],'image_size':meta['image_size'],'source':'independent cloud segmentation boxes; actual masks remain SAM2','frames':frames})

def generate(source_dir,output,c,mirror_model,seed_dataset=None,seed_native=None):
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for source in sorted(source_dir.glob('*.mp4')):
        folder=output/'clips'/source.stem;reuse=False
        if seed_dataset and source.name==json.loads((seed_dataset/'record.json').read_text()).get('name'):
            reuse=sha256(source)==sha256(seed_dataset/'original.video')
        prepare(source,folder,seed_dataset/'source.mp4' if reuse else None)
        row={'video':source.name,'folder':str(folder.resolve()),'status':'generating','original_sha256':sha256(source),'started_at':now()};rows.append(row);write(output/'batch_manifest.json',{'started_at':rows[0]['started_at'],'clips':rows,'status':'running'})
        work=folder/'attempts/0001/work';work.mkdir(parents=True,exist_ok=True)
        try:
            remote=work/'remote_job.json'
            if reuse and seed_native and (seed_native/'remote_job.json').exists() and not remote.exists():shutil.copy2(seed_native/'remote_job.json',remote)
            if remote.exists():
                job=json.loads(remote.read_text());
                if job['source_sha256']!=sha256(folder/'source.mp4'):raise ValueError('已有 GPU job 不属于本视频')
                verified=work/'remote_manifest.json'
                complete=(work/'reconstruction.npz').exists() and verified.exists()
                if complete:
                    prior=json.loads(verified.read_text());complete=prior.get('status')=='ready' and prior['source_sha256']==job['source_sha256'] and sha256(work/'reconstruction.npz')==prior['artifact_sha256']['reconstruction.npz']
                if not complete:collect(folder/'source.mp4',work,c,job['job_id'],poll=10)
            else:run(folder/'source.mp4',work,c,poll=10)
            row.update(status='downloaded',remote_job=json.loads(remote.read_text())['job_id'],archive_sha256=sha256(work/'reconstruction.npz'))
            if not (folder/'result').exists():package(folder/'source.mp4',work/'reconstruction.npz',folder/'result')
            evidence(folder)
            if not (folder/'result/mirror_calibration_report.json').exists():
                with (folder/'mirror.log').open('w') as log:
                    result=subprocess.run([sys.executable,'-B',str(ROOT/'viewer/video_import/estimate_mirror.py'),'--dataset',str(folder),'--model',str(mirror_model),'--device','mps'],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                row['mirror_status']='ready' if result.returncode==0 else 'needs_review'
            else:row['mirror_status']='ready'
            row.update(status='ready',finished_at=now());record=json.loads((folder/'record.json').read_text());write(folder/'record.json',{**record,'status':'ready','viewer':'result/viewer.html'})
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__,error=str(error)[:240],finished_at=now());print('Clip failed',source.name,str(error)[:240],flush=True)
        write(output/'batch_manifest.json',{'started_at':rows[0]['started_at'],'clips':rows,'status':'running'})
        print('Clip',source.name,row['status'],flush=True)
    write(output/'batch_manifest.json',{'started_at':rows[0]['started_at'] if rows else now(),'finished_at':now(),'clips':rows,'status':'ready' if rows and all(r['status']=='ready' for r in rows) else 'incomplete'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--mirror-model',type=Path,required=True);p.add_argument('--seed-dataset',type=Path);p.add_argument('--seed-native',type=Path);a=p.parse_args()
    generate(a.source_dir,a.output,config(a.config),a.mirror_model,a.seed_dataset,a.seed_native)
