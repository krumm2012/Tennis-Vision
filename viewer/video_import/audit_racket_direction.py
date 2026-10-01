"""Replay fixed current-video keypoints; measure shaft direction and head-center errors."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sam3d'))
from fit_racket_pose import project

def audit(poses,keypoints,meta):
 d=json.loads(Path(poses).read_text());k=json.loads(Path(keypoints).read_text());m=json.loads(Path(meta).read_text())
 if d['video_sha256']!=k['video_sha256'] or m['video_sha256']!=k['video_sha256'] or len(d['frames'])!=len(k['frames']):raise ValueError('方向审计数据不一致')
 image=[];center=[];stereo=[];angles=[];scale=m['image_size'][0]/1280
 for i,row in enumerate(d['frames']):
  source=k['frames'][i]
  if row['status']!='fitted' or not source.get('real'):continue
  R=np.array(row['rotation_camera_columns']);t=np.array(row['translation_camera_m']);grip=np.array(row['grip_target_camera_m']);head=R@np.array([0,d['model']['head_center_y_m'],0])+t
  uv=project(np.array([grip,head]),m['focal'][i],m['image_size']);a=uv[1]-uv[0];b=np.array(source['real']['points']['head_center'])-uv[0]
  angle=float(np.degrees(np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8),-1,1))));image.append(angle);center.append(float(np.linalg.norm(uv[1]-source['real']['points']['head_center'])/scale));angles.append(row.get('grip_axis_error_deg',0))
  if source.get('stereo_shaft'):stereo.append(float(np.degrees(np.arccos(np.clip(R[:,1]@np.array(source['stereo_shaft']['direction_camera']),-1,1)))))
 def stats(values):return {'samples':len(values),'median':float(np.median(values)),'p95':float(np.percentile(values,95))} if values else {'samples':0,'median':None,'p95':None}
 return {'image_shaft_error_deg':stats(image),'head_center_error_canonical_px':stats(center),'stereo_shaft_error_deg':stats(stereo),'hand_prior_error_deg':stats(angles),'independent_3d_accuracy':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('poses',type=Path);p.add_argument('--keypoints',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();r=audit(a.poses,a.keypoints,a.meta);print(json.dumps(r,ensure_ascii=False))
 if a.check and (r['image_shaft_error_deg']['p95'] is None or r['image_shaft_error_deg']['p95']>25 or r['head_center_error_canonical_px']['p95']>40 or (r['stereo_shaft_error_deg']['samples'] and r['stereo_shaft_error_deg']['median']>35)):raise SystemExit('FAIL: 拍柄方向/拍头位置与当前视频观测冲突')
