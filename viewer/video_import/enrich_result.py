"""Generate current-video racket evidence and mirror person candidates; no old poses."""
import argparse,json,sys,shutil
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'viewer/sam3d'))
from fit_racket_video import fit_shape,head_ellipse,ring_points
from fit_racket_pose import project,model_points,PARTS
from run_records import write,sha256,now,revision

def enrich(folder,model_path,device='mps',hand='right',result=None):
    from ultralytics import YOLO
    folder=Path(folder);out=Path(result) if result else folder/'result';meta=json.loads((out/'mesh_meta.json').read_text());attempt=json.loads((folder/'record.json').read_text())['attempt']
    archive=folder/'attempts'/f'{attempt:04d}'/'work/reconstruction.npz'
    with np.load(archive,allow_pickle=False) as d:joints=d['joints'].copy();roots=d['source_roots'].copy();focals=d['focal'].copy()
    model=json.loads((ROOT/'output/sam3d_cloud/wilson_model.json').read_text())['model'];detector=YOLO(str(model_path));size=meta['image_size'];count=meta['frames'];fps=meta['fps'];wrist_id=41 if hand=='right' else 62
    cap=cv2.VideoCapture(str(folder/'source.mp4'));people=[];rows=[];previous=None
    for n in range(count):
        ok,image=cap.read()
        if not ok:raise ValueError('视频帧数与姿态不一致')
        wrist=joints[n,wrist_id]+roots[n];wuv=project(wrist[None,:],focals[n],size)[0]
        found=detector.predict(image,classes=[0,38],conf=.015,imgsz=1280,device=device,verbose=False)[0]
        persons=[];candidates=[]
        if found.masks is not None:
            for k,box in enumerate(found.boxes):
                polygon=found.masks.xy[k];cls=int(box.cls[0]);confidence=float(box.conf[0]);xyxy=box.xyxy[0].cpu().numpy()
                if cls==0:
                    if confidence<.15 or np.prod(xyxy[2:]-xyxy[:2])<300*(size[0]/1280)**2:continue
                    simplified=cv2.approxPolyDP(polygon.astype(np.float32),.75*size[0]/1280,True).reshape(-1,2)
                    persons.append({'box':xyxy.tolist(),'polygon':simplified.round(1).tolist(),'confidence':confidence})
                elif cls==38:
                    distance=float(np.linalg.norm(np.maximum(np.maximum(xyxy[:2]-wuv,wuv-xyxy[2:]),0)))
                    if distance<75*size[0]/1280 and confidence>=.04:
                        ellipse=head_ellipse(polygon)
                        if ellipse is not None:candidates.append((confidence*np.exp(-distance/30),confidence,ellipse,polygon))
        people.append({'frame':n,'persons':persons});row={'frame':n,'status':'missing_observation','review_reasons':['missing_observation']}
        if candidates:
            _,confidence,ellipse,polygon=max(candidates,key=lambda c:c[0])
            result=fit_shape(ellipse,wrist,wuv,focals[n],model,previous,image_size=size)
            _,matrix,t,error,gap,distance=result
            if error<=5 and gap<=.2 and distance<=12 and t[2]>.1:
                previous=matrix;row.update(status='fitted',quality='silhouette_fitted',ambiguous=True,source='current_video_silhouette',translation_camera_m=t.tolist(),rotation_camera_columns=matrix.tolist(),mask_fit_rms_px=error,detection_confidence=confidence,review_reasons=['mono_plane_ambiguity'],observed_polygon=polygon.round(1).tolist())
        rows.append(row)
        if n%25==24 or n==count-1:print(f'{n+1}/{count}: fitted {sum(r["status"]=="fitted" for r in rows)}',flush=True)
    cap.release()
    # Interpolate only brief bounded gaps with reliable observations on both sides.
    keys=[r['frame'] for r in rows if r['status']=='fitted'];observed=len(keys)
    for a,b in zip(keys,keys[1:]):
        if 1<b-a<=round(.24*fps)+1:
            rot=Slerp([a,b],Rotation.from_matrix([rows[a]['rotation_camera_columns'],rows[b]['rotation_camera_columns']]))
            grip=np.array([0,model['grip_y_m'],0]);offsets=[np.array(rows[k]['translation_camera_m'])+np.array(rows[k]['rotation_camera_columns'])@grip-(joints[k,wrist_id]+roots[k]) for k in [a,b]]
            for n in range(a+1,b):
                matrix=rot([n]).as_matrix()[0];alpha=(n-a)/(b-a);t=joints[n,wrist_id]+roots[n]+offsets[0]*(1-alpha)+offsets[1]*alpha-matrix@grip
                rows[n].update(status='fitted',quality='interpolated',source='bounded_gap_interpolation',ambiguous=True,translation_camera_m=t.tolist(),rotation_camera_columns=matrix.tolist(),review_reasons=['interpolated','mono_plane_ambiguity'])
    for row in rows:
        if row['status']=='fitted':
            n=row['frame'];matrix=np.array(row['rotation_camera_columns']);t=np.array(row['translation_camera_m']);uv=project(model_points(model)@matrix.T+t,focals[n],size)
            row['projected_points']={name:uv[k].tolist() for k,name in enumerate(PARTS)};row['projected_head_outline']=project(ring_points(model)@matrix.T+t,focals[n],size).tolist()
    summary={'frames':count,'observed':observed,'interpolated':sum(r.get('quality')=='interpolated' for r in rows),'hidden':sum(r['status']!='fitted' for r in rows),'hand':hand,'mono_plane_ambiguity':True}
    payload={'version':1,'video_id':folder.name,'video_sha256':meta['video_sha256'],'fps':fps,'image_size':size,'model':model,'summary':summary,'frames':rows}
    shutil.copy2(ROOT/'output/sam3d_cloud/wilson_mesh.bin',out/'wilson_mesh.bin');write(out/'wilson_model.json',{'model':model});write(out/'person_candidates.json',{'video_sha256':meta['video_sha256'],'image_size':size,'frames':people});write(out/'racket_poses.json',payload)
    write(out/'enrichment_manifest.json',{'video_sha256':meta['video_sha256'],'detector_sha256':sha256(model_path),'model_mesh_sha256':sha256(out/'wilson_mesh.bin'),'racket_poses_sha256':sha256(out/'racket_poses.json'),'person_candidates_sha256':sha256(out/'person_candidates.json'),'git_commit':revision(ROOT),'code_sha256':sha256(Path(__file__)),'finished_at':now(),'summary':summary})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);p.add_argument('--model',type=Path,required=True);p.add_argument('--device',default='mps');p.add_argument('--hand',choices=['right','left'],default='right');a=p.parse_args();enrich(a.dataset,a.model,a.device,a.hand,a.result)
