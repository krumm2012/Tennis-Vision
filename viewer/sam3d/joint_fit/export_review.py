"""Export a diagnostic joint/racket comparison; no claim of a rigged hand mesh."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('output/sam3d_cloud'));args=ap.parse_args();root=args.data;p=root/'joint_fit_v4';d=json.loads((p/'viewer_data.json').read_text());model=d['model'];cap=cv2.VideoCapture(str(root/'video.mp4'));rows=d['frames'];dt=np.diff([r['time'] for r in rows]);fps=1/np.median(dt)
    if not np.allclose(dt,1/fps,atol=1e-5):raise ValueError('Review export requires uniform output timebase')
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','2000x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(p/'joint_review.mp4')],stdin=subprocess.PIPE)
    thumbs=[];indices=set(np.linspace(0,len(rows)-1,6).astype(int));chains=[[20,t+3,t+2,t+1,t] for t in [0,4,8,12,16]]
    def line(im,points,col,thick=2,closed=False):cv2.polylines(im,[np.round(points).astype(np.int32)],closed,col,thick,cv2.LINE_AA)
    try:
        for i,r in enumerate(rows):
            ok,im=cap.read()
            if not ok:raise ValueError('Missing source frame')
            uv=np.array(r['hand_image']);ringuv=np.array(r['ring_image']);line(im,ringuv,(190,220,100),2,True)
            for c in chains:line(im,uv[c],(90,180,255))
            cv2.rectangle(im,(0,0),(1280,62),(25,25,25),-1);cv2.putText(im,f'JOINT CANDIDATE | FRAME {i+1:03d} | OBSERVATIONS, NOT GROUND TRUTH',(18,40),cv2.FONT_HERSHEY_SIMPLEX,.72,(240,240,240),2,cv2.LINE_AA)
            panel=np.full((720,720,3),(38,29,20),np.uint8);R=np.array(r['rotation_camera_columns']);g=np.array(r['grip_camera_m']);ring=(np.array(model['head_outline'])-[0,model['grip_y_m'],0])@R.T+g;h=np.array(r['joints_camera'])[21:42];center=(h[20]+ring.mean(0))*.5
            yaw=-.6;pitch=.6;q=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]]);v=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
            def proj(x):return ((np.asarray(x)-center)@q.T@v.T)[:,:2]*750+[360,360]
            line(panel,proj(ring),(190,220,100),3,True);shaft=(np.array([[0,0,0],[0,model['throat_y_m'],0]])-[0,model['grip_y_m'],0])@R.T+g;line(panel,proj(shaft),(235,235,235),7)
            for c in chains:line(panel,proj(h[c]),(90,180,255),3)
            cv2.putText(panel,'3D JOINTS | HAND MESH UNCHANGED',(18,38),cv2.FONT_HERSHEY_SIMPLEX,.66,(240,240,240),2)
            cv2.putText(panel,f'Contour {r["metrics"]["contour_rms_px"]:.1f}px | Contact proxy {r["metrics"]["contact_proxy_mm"]:.1f}mm',(18,666),cv2.FONT_HERSHEY_SIMPLEX,.58,(220,220,220),1)
            cv2.putText(panel,'REVIEW REQUIRED' if r['review_reasons'] else 'MODEL CHECKS ONLY',(18,698),cv2.FONT_HERSHEY_SIMPLEX,.65,(100,190,245),1)
            frame=np.hstack((im,panel));proc.stdin.write(frame.tobytes())
            if i in indices:thumbs.append(cv2.resize(frame,(1000,360)))
    finally:
        cap.release();proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError('Video encoder failed')
    cv2.imwrite(str(p/'overview.jpg'),np.vstack(thumbs));print(p/'joint_review.mp4')
if __name__=='__main__':main()
