"""Cloud-only mirror SAM3D and genuine two-object SAM2 video propagation.

Mirror SAM is inferred on a horizontally flipped full image, then mapped back
into the original virtual-camera axes. All native raw outputs are retained.
"""
import argparse,json,os,sys,time,gc
from pathlib import Path
import cv2,numpy as np
from run_records import write,sha256,now
from mhr_parameters import capture,stack

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1))

def choose_roles(boxes,confidence,real_box,previous_mirror=None):
    real=max(range(len(boxes)),key=lambda i:iou(boxes[i],real_box))
    rb=boxes[real];rc=(rb[:2]+rb[2:])/2;rh=rb[3]-rb[1];options=[]
    for i,b in enumerate(boxes):
        bc=(b[:2]+b[2:])/2;bh=b[3]-b[1]
        if i==real or confidence[i]<.35 or iou(b,rb)>.25 or not .25*rh<bh<2*rh:continue
        if bc[1]<rc[1]-.25*rh and abs(bc[0]-rc[0])<3*rh:
            score=float(confidence[i])+(iou(b,previous_mirror) if previous_mirror is not None else 0)
            options.append((score,i))
    return real,max(options)[1] if options else None

def run(video,output):
    import torch
    from ultralytics import YOLO
    started=time.monotonic();video=Path(video);out=Path(output);checkpoint=Path(os.environ['VIEWER_SAM2_WEIGHTS'])
    if not torch.cuda.is_available() or not checkpoint.is_file():raise RuntimeError('需要 CUDA 与 SAM2 权重')
    data={}
    with np.load(out/'reconstruction.npz',allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
    count=len(data['vertices']);cap=cv2.VideoCapture(str(video));size=np.array([int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]);fps=cap.get(cv2.CAP_PROP_FPS)
    detector=YOLO(os.environ['VIEWER_SEGMENTATION_WEIGHTS']);rows=[];previous=None;polygons=[]
    for frame in range(count):
        ok,image=cap.read()
        if not ok:raise ValueError('视频与原人体结果帧数不一致')
        p=data['vertices'][frame]+data['source_roots'][frame];uv=p[:,:2]/p[:,2:]*data['focal'][frame]+size/2;real_box=np.r_[uv.min(0),uv.max(0)]
        found=detector.predict(image,classes=[0],verbose=False)[0]
        if found.masks is None:raise ValueError('双人物提示检测失败')
        boxes=found.boxes.xyxy.cpu().numpy();confidence=found.boxes.conf.cpu().numpy();r,m=choose_roles(boxes,confidence,real_box,previous)
        if iou(boxes[r],real_box)<.3:raise ValueError('真人检测与已有 SAM 身份不匹配')
        row={'frame':frame,'real_box':boxes[r].tolist(),'real_confidence':float(confidence[r]),'mirror_box':boxes[m].tolist() if m is not None else None,'mirror_confidence':float(confidence[m]) if m is not None else 0}
        if m is not None:previous=boxes[m]
        rows.append(row);polygons.append([found.masks.xy[r].copy(),found.masks.xy[m].copy() if m is not None else None])
    cap.release();del detector;torch.cuda.empty_cache()
    valid=np.array([r['mirror_box'] is not None for r in rows]);mirror_found=bool(valid.mean()>=.5)
    data['mirror_valid']=valid if mirror_found else np.zeros(count,bool)
    if mirror_found:
        sys.path.insert(0,os.environ['SAM3D_BODY_CODE'])
        from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
        weights=Path(os.environ['SAM3D_WEIGHTS']);model,cfg=load_sam_3d_body(str(weights/'model.ckpt'),device='cuda',mhr_path=str(weights/'assets/mhr_model.pt'));model.eval();est=SAM3DBodyEstimator(model,cfg)
        mv=np.zeros_like(data['vertices']);mj=np.zeros_like(data['joints']);mr=np.zeros_like(data['source_roots']);mf=np.zeros_like(data['focal']);muv=np.zeros_like(data['joints2d']);cap=cv2.VideoCapture(str(video));native=[None]*count
        for frame in range(count):
            ok,image=cap.read()
            if not ok:raise ValueError('镜中 SAM 输入帧缺失')
            if not valid[frame]:continue
            box=np.asarray(rows[frame]['mirror_box']);flipped_box=box.copy();flipped_box[[0,2]]=size[0]-box[[2,0]]
            with torch.no_grad():result=est.process_one_image(cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB),bboxes=flipped_box[None].astype(np.float32),inference_type='full')
            if not result:raise ValueError('镜中 SAM 未返回网格')
            person=result[0];native[frame]=capture(person);mv[frame]=person['pred_vertices'];mj[frame]=person['pred_keypoints_3d'];mr[frame]=np.asarray(person['pred_cam_t']).reshape(3);mf[frame]=float(np.asarray(person['focal_length']).reshape(-1)[0]);muv[frame]=person['pred_keypoints_2d']
            mv[frame,:,0]*=-1;mj[frame,:,0]*=-1;mr[frame,0]*=-1;muv[frame,:,0]=size[0]-muv[frame,:,0]
            if frame%25==24 or frame==count-1:print(f'Mirror SAM3D {frame+1}/{count}',flush=True)
        cap.release();data.update(mirror_vertices=mv,mirror_joints=mj,mirror_roots=mr,mirror_focal=mf,mirror_joints2d=muv);data.update(stack(native,'mirror_'))
        del est,model;gc.collect();torch.cuda.empty_cache()
    # Fixed crop derived from this clip's tracked boxes gives both people useful encoder pixels.
    all_boxes=np.array([box for r in rows for box in [r['real_box'],r['mirror_box']] if box is not None]);lo=all_boxes[:,:2].min(0);hi=all_boxes[:,2:].max(0);margin=np.maximum((hi-lo)*.12,20);lo=np.maximum(0,np.floor(lo-margin)).astype(int);hi=np.minimum(size,np.ceil(hi+margin)).astype(int)
    x0,y0=lo;x1,y1=hi;images=out/'sam2_frames';images.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(video))
    for frame in range(count):
        ok,image=cap.read()
        if not ok:raise ValueError('SAM2 视频缺帧')
        if not cv2.imwrite(str(images/f'{frame:05d}.jpg'),image[y0:y1,x0:x1],[cv2.IMWRITE_JPEG_QUALITY,95]):raise ValueError('SAM2 临时帧写入失败')
    cap.release()
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',str(checkpoint),device='cuda',apply_postprocessing=False)
    seeds=[];mask_width=512;mask_height=round(mask_width*size[1]/size[0]);real_masks=np.zeros((count,mask_height,mask_width),np.uint8);mirror_masks=np.zeros_like(real_masks);reports=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=predictor.init_state(str(images),offload_video_to_cpu=True,offload_state_to_cpu=True)
        for obj in [1,2] if mirror_found else [1]:
            for start in range(0,count,25):
                candidates=[i for i in range(start,min(start+25,count)) if polygons[i][obj-1] is not None and rows[i]['real_confidence' if obj==1 else 'mirror_confidence']>=.55]
                if not candidates:continue
                frame=0 if start==0 and polygons[0][obj-1] is not None else max(candidates,key=lambda i:rows[i]['real_confidence' if obj==1 else 'mirror_confidence'])
                mask=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillPoly(mask,[np.round(polygons[frame][obj-1]-lo).astype(np.int32)],1);predictor.add_new_mask(state,frame_idx=frame,obj_id=obj,mask=mask);seeds.append({'frame':frame,'object':obj})
        for frame,ids,logits in predictor.propagate_in_video(state,start_frame_idx=0):
            values=logits[:,0].float().cpu().numpy();raw={int(obj):values[k] for k,obj in enumerate(ids)}
            if 1 in raw and 2 in raw:
                overlap=(raw[1]>0)&(raw[2]>0);prefer_real=raw[1]>=raw[2];raw[1]=np.where(overlap&~prefer_real,-20,raw[1]);raw[2]=np.where(overlap&prefer_real,-20,raw[2])
            quality={'frame':int(frame)}
            for obj in [1,2]:
                canvas=np.zeros((int(size[1]),int(size[0])),np.uint8);logit=raw.get(obj)
                if logit is not None:
                    canvas[y0:y1,x0:x1]=(255/(1+np.exp(-np.clip(logit,-20,20)))).astype(np.uint8)
                tile=cv2.resize(canvas,(mask_width,mask_height),interpolation=cv2.INTER_AREA);(real_masks if obj==1 else mirror_masks)[frame]=tile
                selected=tile>140;poly=polygons[frame][obj-1];reference=np.zeros_like(tile)
                if poly is not None:cv2.fillPoly(reference,[np.round(poly/size*[mask_width,mask_height]).astype(np.int32)],1)
                intersection=(selected&(reference>0)).sum();union=(selected|(reference>0)).sum();quality['real_iou' if obj==1 else 'mirror_iou']=float(intersection/max(union,1));quality['real_area' if obj==1 else 'mirror_area']=int(selected.sum())
            reports.append(quality)
            if frame%25==24 or frame==count-1:print(f'SAM2 dual tracking {frame+1}/{count}',flush=True)
    if sorted(r['frame'] for r in reports)!=list(range(count)):raise ValueError('SAM2 未覆盖完整时间轴')
    if any(r['real_area']==0 for r in reports):raise ValueError('SAM2 真人跟踪丢失')
    data['masks']=real_masks;data['masks_mirror_sam2']=mirror_masks
    np.savez_compressed(out/'reconstruction.npz',**data)
    report={'source_sha256':sha256(video),'frames':count,'image_size':size.tolist(),'fps':fps,'mirror_available':mirror_found,'mirror_sam3d_frames':int(data['mirror_valid'].sum()),'mirror_mhr_parameters_available':'mirror_mhr_model_params' in data,'mirror_sam3d_method':'independent inference on horizontal-flipped image; x restored to original virtual camera; same anatomical MHR IDs','sam2_model':'SAM2.1_hiera_small','sam2_checkpoint_sha256':sha256(checkpoint),'sam2_crop':[*lo.tolist(),*hi.tolist()],'sam2_seeds':seeds,'sam2_mask_size':[mask_width,mask_height],'sam2_objects':[1,2] if mirror_found else [1],'tracking':rows,'mask_quality':reports,'seconds':time.monotonic()-started,'finished_at':now(),'sam2_real_iou_median':float(np.median([r['real_iou'] for r in reports])),'sam2_mirror_iou_median':float(np.median([r['mirror_iou'] for r in reports]))}
    write(out/'multiview_manifest.json',report)
    for file in images.glob('*.jpg'):file.unlink()
    images.rmdir();print(json.dumps({k:v for k,v in report.items() if k not in ['tracking','mask_quality','sam2_seeds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.video,a.output)
