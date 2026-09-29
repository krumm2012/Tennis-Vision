"""Match a transcoded sequence to decoded native frames using PTS and image evidence."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np

def load(path):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
    times=np.array([float(f['best_effort_timestamp_time']) for f in info['frames']]);cap=cv2.VideoCapture(str(path));images=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        images.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(160,90)).astype(np.float32))
    cap.release();assert len(images)==len(times)
    return times,np.stack(images)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',type=Path,required=True);ap.add_argument('--viewer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    nt,ni=load(a.native);vt,vi=load(a.viewer);rows=[]
    for i,t in enumerate(vt):
        candidates=np.flatnonzero(abs(nt-t)<=.085)
        if not len(candidates):raise ValueError(f'No native frame near viewer frame {i}')
        errors=np.abs(ni[candidates]-vi[i]).mean(axis=(1,2));order=np.argsort(errors);k=int(candidates[order[0]])
        rows.append({'viewer_frame':i,'native_frame':k,'viewer_time':float(t),'native_time':float(nt[k]),'image_mae':float(errors[order[0]]),'runner_up_margin':float(errors[order[1]]-errors[order[0]]) if len(order)>1 else None})
    assert all(b['native_frame']>=a['native_frame'] for a,b in zip(rows,rows[1:])), 'Nonmonotonic matches need review'
    a.output.write_text(json.dumps({'method':'minimum grayscale MAE among PTS neighbors within 85ms; estimated correspondence','frames':rows},indent=2))
if __name__=='__main__':main()
