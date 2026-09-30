"""Local SAM + instance segmentation runner for the video-library contract.

Configure SAM3D_BODY_CODE, SAM3D_WEIGHTS, SAM3D_FACES (matching .npy topology),
and VIEWER_SEGMENTATION_WEIGHTS (local Ultralytics segmentation model).
Tracks one person by overlap after choosing the largest initial detection.
"""
import argparse, os, sys
from pathlib import Path

def requirements():
    checks={'SAM3D_BODY_CODE':'dir','SAM3D_WEIGHTS':'dir','SAM3D_FACES':'file','VIEWER_SEGMENTATION_WEIGHTS':'file'}
    missing=[]
    for key,kind in checks.items():
        value=os.environ.get(key,'');path=Path(value)
        if not value or not (path.is_dir() if kind=='dir' else path.is_file()):missing.append(key)
    if not missing:
        weights=Path(os.environ['SAM3D_WEIGHTS'])
        for name in ('model.ckpt','assets/mhr_model.pt'):
            if not (weights/name).is_file():missing.append('SAM3D_WEIGHTS/'+name)
    return missing

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    missing=requirements()
    if missing:raise RuntimeError('缺少模型配置：'+', '.join(missing))
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    sys.path.insert(0,os.environ['SAM3D_BODY_CODE'])
    from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
    weights=Path(os.environ['SAM3D_WEIGHTS']);device='cuda' if torch.cuda.is_available() else 'cpu'
    model,cfg=load_sam_3d_body(str(weights/'model.ckpt'),device=device,mhr_path=str(weights/'assets/mhr_model.pt'));model.eval();est=SAM3DBodyEstimator(model,cfg)
    segment=YOLO(os.environ['VIEWER_SEGMENTATION_WEIGHTS']);faces=np.load(os.environ['SAM3D_FACES'],allow_pickle=False)
    cap=cv2.VideoCapture(str(args.video));vertices=[];roots=[];focals=[];masks=[];previous=None
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            detection=segment.predict(image,classes=[0],verbose=False)[0]
            if detection.masks is None or len(detection.boxes)==0:raise ValueError(f'第 {len(vertices)+1} 帧未找到人物；请选用人物清晰的片段')
            boxes=detection.boxes.xyxy.cpu().numpy()
            if previous is None:index=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
            else:
                lo=np.maximum(boxes[:,:2],previous[:2]);hi=np.minimum(boxes[:,2:],previous[2:]);inter=np.maximum(hi-lo,0).prod(axis=1);area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);old=(previous[2]-previous[0])*(previous[3]-previous[1]);iou=inter/(area+old-inter+1e-9);index=int(np.argmax(iou))
                if iou[index]<.1:raise ValueError('人物跟踪中断；请使用连续、单一主体的片段')
            previous=boxes[index]
            with torch.no_grad():result=est.process_one_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),bboxes=previous[None].astype(np.float32),inference_type='full')
            if not result:raise ValueError('SAM 未生成人体网格')
            person=result[0];vertices.append(np.asarray(person['pred_vertices'],dtype=np.float32));roots.append(np.asarray(person['pred_cam_t'],dtype=np.float32).reshape(3));focals.append(float(np.asarray(person['focal_length']).reshape(-1)[0]))
            # Polygons are in original image coordinates; avoid resizing letterboxed masks.
            mask=np.zeros(image.shape[:2],np.uint8);polygon=detection.masks.xy[index].astype(np.int32)
            if len(polygon)<3:raise ValueError('人物遮罩为空')
            cv2.fillPoly(mask,[polygon],255);scale=320/max(mask.shape);masks.append(cv2.resize(mask,(max(1,round(mask.shape[1]*scale)),max(1,round(mask.shape[0]*scale))),interpolation=cv2.INTER_AREA))
            print(f'Generated {len(vertices)} frames',flush=True)
    finally:cap.release()
    if not vertices:raise ValueError('视频没有可解码的画面')
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'reconstruction.npz',vertices=vertices,faces=faces,source_roots=roots,focal=focals,masks=masks)

if __name__=='__main__':main()
