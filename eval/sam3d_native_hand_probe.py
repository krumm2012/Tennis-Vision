"""Run native-resolution SAM full inference without modifying existing Viewer data.

Cloud prerequisites: input_native.mp4 (2560x1440), mesh_pose.json person boxes
from the 1280x720 sequence, and existing SAM weights/venv. Native frame indexes
are decoded frame indexes, NOT Viewer indexes: apply map_native_video_frames.py.
The inherited person box can be offset by one frame after transcoding; its padding
is retained. Record model hand gates; never force rejected hand predictions.
"""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/root/tennis-sam3d'));ap.add_argument('--frames',type=int,nargs='*',default=list(range(250)));args=ap.parse_args()
ROOT=args.root;sys.path.insert(0,str(ROOT/'sam-3d-body'))
from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
outdir=ROOT/'native_hand_sequence';outdir.mkdir(exist_ok=True)
w=ROOT/'weights';model,cfg=load_sam_3d_body(str(w/'model.ckpt'),device='cuda',mhr_path=str(w/'assets/mhr_model.pt'));est=SAM3DBodyEstimator(model,cfg)
refs=json.loads((ROOT/'mesh_pose.json').read_text())['frames'];cap=cv2.VideoCapture(str(ROOT/'input_native.mp4'))
results=[];vertices=[];joints=[];joints2d=[];translations=[];focals=[];params=[]
for i in args.frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok
 box=np.array(refs[i]['bbox'],dtype=np.float32)
 assert im.shape[:2]==(1440,2560), 'Expected native 2560x1440 input'
 box*=2
 for mode in ['full']:
  start=time.time()
  diagnostics={}
  def capture(frame,event,arg):
   if event=='return' and frame.f_code.co_name=='run_inference':
    for key in ['angle_difference_valid_mask','hand_box_size_valid_mask','hand_kps2d_valid_mask','hand_wrist_kps2d_valid_mask','hand_valid_mask','valid_angle']:
     v=frame.f_locals.get(key)
     if isinstance(v,torch.Tensor):diagnostics[key]=v.detach().cpu().tolist()
  sys.setprofile(capture)
  with torch.no_grad():p=est.process_one_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),bboxes=box[None],inference_type=mode)[0]
  sys.setprofile(None)
  results.append({'frame':i,'diagnostics':diagnostics,'rhand_bbox':p['rhand_bbox'].tolist(),'lhand_bbox':p['lhand_bbox'].tolist()})
  vertices.append(p['pred_vertices']);joints.append(p['pred_keypoints_3d']);joints2d.append(p['pred_keypoints_2d']);translations.append(p['pred_cam_t']);focals.append(p['focal_length']);params.append(p['hand_pose_params'])
  print(i,mode,'seconds',round(time.time()-start,3),'accepted',diagnostics.get('valid_angle'),flush=True)
np.savez_compressed(outdir/'sequence.npz',vertices=vertices,joints=joints,joints2d=joints2d,translations=translations,focals=focals,hand_params=params,frame_indices=args.frames)
(outdir/'diagnostics.json').write_text(json.dumps({'image_size':[2560,1440],'inference_type':'full','frames':results},indent=2))
print('COMPLETE',flush=True)
