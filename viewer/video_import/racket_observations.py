"""Original-resolution hand/racket crops on the existing SAM playback timeline."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'viewer/sam3d'))
from fit_racket_pose import project
from run_records import write,sha256,now

def crop_region(center,radius,size):
    w,h=size;x,y=np.asarray(center)
    return (max(0,int(x-radius)),max(0,int(y-radius)),min(w,int(x+radius)),min(h,int(y+radius)))

def detect(folder,model_path,device='mps',result=None):
    from ultralytics import YOLO
    folder=Path(folder);out=Path(result) if result else folder/'result';meta=json.loads((out/'mesh_meta.json').read_text());attempt=json.loads((folder/'record.json').read_text())['attempt']
    with np.load(folder/'attempts'/f'{attempt:04d}'/'work/reconstruction.npz',allow_pickle=False) as d:joints=d['joints'];roots=d['source_roots'];focal=d['focal']
    source=folder/'original.video' if (folder/'original.video').exists() else folder/'source.mp4'
    cap=cv2.VideoCapture(str(source));rate=cap.get(cv2.CAP_PROP_FPS);detector=YOLO(str(model_path));size=np.array(meta['image_size']);previous_index=-1;frames=[]
    mirror=json.loads((out/'mirror_geometry.json').read_text()) if meta.get('mirror_available') else None
    for n in range(meta['frames']):
        index=int(round(n/meta['fps']*rate))
        if index!=previous_index+1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,image=cap.read();previous_index=index
        if not ok:raise ValueError('原视频时间轴无法与 SAM 帧对应')
        original_size=np.array([image.shape[1],image.shape[0]]);scale=original_size/size
        wrist=joints[n,41]+roots[n];centers=[('real',wrist)]
        if mirror:
            normal=np.array(mirror['normal_camera']);normal/=np.linalg.norm(normal);plane=mirror['distance_camera_m'];centers.append(('mirror',wrist-2*(wrist@normal-plane)*normal))
        candidates=[]
        for view,point in centers:
            uv=project(point[None,:],focal[n],size)[0];radius=np.clip(focal[n]/max(point[2],.1)*.95,100*size[0]/1280,230*size[0]/1280)
            x0,y0,x1,y1=crop_region(uv*scale,radius*max(scale),original_size)
            if x1-x0<80 or y1-y0<80:continue
            found=detector.predict(image[y0:y1,x0:x1],classes=[38],conf=.035,imgsz=960,device=device,verbose=False)[0]
            if found.masks is None:continue
            for k,box in enumerate(found.boxes):
                polygon=(found.masks.xy[k]+[x0,y0])/scale;confidence=float(box.conf[0]);xyxy=box.xyxy[0].cpu().numpy();xyxy=(xyxy+np.tile([x0,y0],2))/np.tile(scale,2)
                candidates.append({'view':view,'box':xyxy.tolist(),'polygon':polygon.round(2).tolist(),'confidence':confidence,'source':'original_resolution_roi'})
        frames.append({'frame':n,'source_frame':index,'candidates':candidates})
        if n%25==24 or n==meta['frames']-1:print(f'{n+1}/{meta["frames"]}: real {sum(any(c["view"]=="real" for c in r["candidates"]) for r in frames)}',flush=True)
    cap.release()
    write(out/'racket_roi_candidates.json',{'video_sha256':meta['video_sha256'],'original_sha256':sha256(source),'original_size':original_size.tolist(),'original_fps':rate,'image_size':size.tolist(),'fps':meta['fps'],'detector_sha256':sha256(model_path),'finished_at':now(),'frames':frames})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);p.add_argument('--model',type=Path,required=True);p.add_argument('--device',default='mps');a=p.parse_args();detect(a.dataset,a.model,a.device,a.result)
