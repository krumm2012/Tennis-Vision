"""Estimate a fixed SAM-camera mirror from this video's reflected COCO joints."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from run_records import write,sha256
from mirror_calibration import apply_geometry

def estimate(folder,model_path,device='mps',reuse=False,result=None):
 from ultralytics import YOLO
 folder=Path(folder);out=Path(result) if result else folder/'result';meta=json.loads((out/'mesh_meta.json').read_text());record=json.loads((folder/'record.json').read_text())
 with np.load(folder/'attempts'/f"{record['attempt']:04d}"/'work/reconstruction.npz',allow_pickle=False) as d:joints=d['joints'].copy();roots=d['source_roots'].copy();vertices=d['vertices'].copy()
 size=np.array(meta['image_size']);focals=np.array(meta['focal']);cached=None
 if reuse:
  source=json.loads((out/'mirror_pose.json').read_text())
  if source['video_sha256']!=meta['video_sha256'] or source['detector_sha256']!=sha256(model_path):raise ValueError('镜中观测缓存不匹配')
  cached=source['frames']
 detector=YOLO(str(model_path)) if not cached else None;sam_ids=[];cap=cv2.VideoCapture(str(folder/'source.mp4'));rows=[];points=[];observed=[];indices=[]
 # Anatomical labels exchange under mirror reflection, as in the default pipeline.
 pairs=[(5,6),(6,5),(7,8),(8,7),(41,9),(62,10),(9,12),(10,11),(11,14),(12,13),(13,16),(14,15)]
 for frame in range(meta['frames']):
  ok,image=cap.read()
  if not ok:raise ValueError('镜像视频帧缺失')
  camera=vertices[frame]+roots[frame];uv=camera[:,:2]/camera[:,2:]*focals[frame]+size/2;real_center=(uv.min(0)+uv.max(0))/2;real_height=np.ptp(uv[:,1])
  choices=[]
  if cached:
   source=cached[frame]
   if source['image'] is not None:choices=[(source['confidence'],np.array(source['image']),np.array(source['box']))]
  else:
   result=detector.predict(image,conf=.25,imgsz=1280,device=device,verbose=False)[0];choices=[]
   if result.keypoints is not None:
    for box,kp in zip(result.boxes,result.keypoints.data.cpu().numpy()):
     b=box.xyxy[0].cpu().numpy();center=(b[:2]+b[2:])/2
     if center[1]<real_center[1]-real_height*.35 and abs(center[0]-real_center[0])<size[0]*.35 and (kp[:,2]>.5).sum()>=8:choices.append((float(box.conf[0]),kp,b))
  row={'frame':frame,'image':None}
  if choices:
   confidence,kp,b=max(choices,key=lambda c:c[0]);row.update(image=kp.tolist(),box=b.tolist(),confidence=confidence)
   for sam,coco in pairs:
    if kp[coco,2]>=.65:points.append(joints[frame,sam]+roots[frame]);observed.append(kp[coco,:2]);indices.append(frame);sam_ids.append(sam)
  rows.append(row)
  if frame%25==24 or frame==meta['frames']-1:print(f'{frame+1}/{meta["frames"]}: mirror observations {sum(r["image"] is not None for r in rows)}',flush=True)
 cap.release();write(out/'mirror_pose.json',{'video_sha256':meta['video_sha256'],'image_size':size.tolist(),'detector_sha256':sha256(model_path),'frames':rows})
 x=np.array(points);obs=np.array(observed);idx=np.array(indices);train=idx%5!=0;core=np.isin(sam_ids,[5,6,9,10,11,12,13,14]);fit_samples=train&core
 if len(points)<60 or train.sum()<40 or (~train).sum()<10:raise ValueError('可靠镜中关节点不足，需手动配对标定')
 def normal(q):return np.array([np.sin(q[0])*np.cos(q[1]),np.sin(q[1]),np.cos(q[0])*np.cos(q[1])])
 def projection(q):
  n=normal(q);y=x-2*(x@n-q[2])[:,None]*n;return y[:,:2]/np.maximum(y[:,2:],.1)*focals[idx,None]+size/2,y
 def residual(q):
  uv,y=projection(q);return np.r_[(uv[fit_samples]-obs[fit_samples]).ravel(),np.minimum(y[fit_samples,2]-.1,0)*100]
 best=None;depth=float(np.median(x[:,2]));lower=[-1.2,-.9,max(.5,depth*.4)];upper=[1.2,.9,depth*4]
 for a in [-.4,0,.4]:
  for b in [-.4,0,.4]:
   fitted=least_squares(residual,[a,b,depth+2],bounds=(lower,upper),loss='soft_l1',f_scale=6,max_nfev=350)
   if best is None or fitted.cost<best.cost:best=fitted
 uv,y=projection(best.x);errors=np.linalg.norm(uv-obs,axis=1);report={'normal_camera':normal(best.x).tolist(),'distance_camera_m':float(best.x[2]),'status':'joint_refined_estimate','video_sha256':meta['video_sha256'],'train_median_px':float(np.median(errors[train])),'heldout_median_px':float(np.median(errors[~train])),'heldout_p90_px':float(np.percentile(errors[~train],90)),'correspondences':len(points),'validation':'every fifth frame excluded from fit','measured_camera':False,'detector_sha256':sha256(model_path)}
 report['core_heldout_median_px']=float(np.median(errors[(~train)&core]));report['core_heldout_p90_px']=float(np.percentile(errors[(~train)&core],90));report['fit_joint_scope']='shoulders_hips_knees_ankles; wrists/elbows reported separately'
 evidence=json.loads((out/'person_candidates.json').read_text())
 if evidence['video_sha256']!=meta['video_sha256']:raise ValueError('镜中遮罩证据与视频不匹配')
 overlaps=[];n=normal(best.x)
 def iou(a,b):
  inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return float(inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1))
 for i in range(0,meta['frames'],5):
  camera=vertices[i]+roots[i];ref=camera-2*(camera@n-best.x[2])[:,None]*n;p=ref[:,:2]/ref[:,2:]*focals[i]+size/2;box=np.r_[p.min(0),p.max(0)];real=camera[:,:2]/camera[:,2:]*focals[i]+size/2;realbox=np.r_[real.min(0),real.max(0)]
  overlaps.append(max((iou(box,np.array(c['box'])) for c in evidence['frames'][i]['persons'] if iou(realbox,np.array(c['box']))<.25),default=0))
 report['heldout_box_iou_median']=float(np.median(overlaps));report['heldout_box_fraction_above_04']=float(np.mean(np.array(overlaps)>=.4));report['texture_gate']='heldout_joint_median_and_independent_person_box_overlap; per_pixel_mask_and_depth';report['joint_outliers_present']=report['core_heldout_p90_px']>30
 write(out/'mirror_joint_fit_report.json',report)
 if report['core_heldout_median_px']>12 or report['heldout_box_iou_median']<.5 or report['heldout_box_fraction_above_04']<.75 or np.min(y[:,2])<=.1:raise ValueError('镜像关节配准未通过，请查看 mirror_joint_fit_report.json')
 report=apply_geometry(folder,report,result=out);print(json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);p.add_argument('--model',type=Path,required=True);p.add_argument('--device',default='mps');p.add_argument('--reuse-observations',action='store_true');a=p.parse_args();estimate(a.dataset,a.model,a.device,a.reuse_observations,a.result)
