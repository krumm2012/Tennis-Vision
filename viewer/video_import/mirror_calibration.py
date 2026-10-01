"""Fit a per-video mirror from paired measured ground corners and rebuild G masks."""
import json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from run_records import write

def corners(value,size):
    points=np.asarray(value['points'],float)
    if points.shape!=(4,2) or not np.isfinite(points).all() or np.any(points<0) or np.any(points>size):raise ValueError('需要四个画面内的有效角点')
    edges=np.roll(points,-1,axis=0)-points;turn=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if not (np.all(turn>1) or np.all(turn< -1)):raise ValueError('角点交叉或共线，请依次选择 ABCD')
    if abs(cv2.contourArea(points.astype(np.float32)))<500:raise ValueError('标定区域太小')
    return points

def fit(bundle,meta):
    if bundle.get('video_sha256')!=meta['video_sha256'] or bundle.get('image_size')!=meta['image_size']:raise ValueError('标定与当前视频不匹配')
    size=np.array(meta['image_size']);g=bundle['ground'];m=bundle['mirror'];ground=corners(g,size);mirror=corners(m,size)
    width=float(g['width']);length=float(g['length'])
    if not .1<=width<=50 or not .1<=length<=50:raise ValueError('请填写有效实测宽长')
    for row in [g,m]:
        if type(row['frame']) is not int or not 0<=row['frame']<meta['frames']:raise ValueError('标定帧号越界')
    world=np.array([[0,0,0],[width,0,0],[width,0,length],[0,0,length]],float);fg=meta['focal'][g['frame']];fm=meta['focal'][m['frame']]
    def uv(x,f):return x[:,:2]/np.maximum(x[:,2:],.1)*f+size/2
    def pnp(points,f):
        camera=np.array([[f,0,size[0]/2],[0,f,size[1]/2],[0,0,1.]])
        ok,r,t,_=cv2.solvePnPGeneric(world,points,camera,None,flags=cv2.SOLVEPNP_IPPE)
        if not ok:raise ValueError('无法初始化地面标定')
        return [(Rotation.from_rotvec(a.ravel()).as_matrix(),b.ravel()) for a,b in zip(r,t)]
    def unpack(q):
        r=Rotation.from_rotvec(q[:3]).as_matrix();n=np.array([np.sin(q[6])*np.cos(q[7]),np.sin(q[7]),np.cos(q[6])*np.cos(q[7])]);x=world@r.T+q[3:6];y=x-2*(x@n-q[8])[:,None]*n;return x,y,n
    def residual(q):
        x,y,_=unpack(q);return np.r_[(uv(x,fg)-ground).ravel(),(uv(y,fm)-mirror).ravel(),np.minimum(x[:,2]-.1,0)*100,np.minimum(y[:,2]-.1,0)*100]
    candidates=[]
    for r,t in pnp(ground,fg):
        x=world@r.T+t
        for rr,tt in pnp(mirror,fm):
            y=world@rr.T+tt;delta=x-y;_,_,v=np.linalg.svd(delta);n=v[0];dist=float(np.mean((x+y)@n/2));start=np.r_[Rotation.from_matrix(r).as_rotvec(),t,np.arctan2(n[0],n[2]),np.arcsin(n[1]),dist]
            solved=least_squares(residual,start,max_nfev=600,loss='soft_l1',f_scale=2);x1,y1,n1=unpack(solved.x);errors=np.r_[np.linalg.norm(uv(x1,fg)-ground,axis=1),np.linalg.norm(uv(y1,fm)-mirror,axis=1)];score=float(np.sqrt(np.mean(errors**2)))
            if min(x1[:,2].min(),y1[:,2].min())>.1:candidates.append((score,solved.x,n1,errors))
    if not candidates:raise ValueError('配对点无法生成正深度镜面')
    score,q,n,errors=min(candidates,key=lambda row:row[0])
    if score>8 or errors.max()>15:raise ValueError(f'配对点不一致：RMS {score:.1f} px，请检查 A↔A′ 对应及尺寸')
    return {'normal_camera':n.tolist(),'distance_camera_m':float(q[8]),'status':'paired_ground_estimate','video_sha256':meta['video_sha256'],'rms_px':score,'corner_errors_px':errors.tolist(),'measured_camera':False,'ground_to_camera_rotation':Rotation.from_rotvec(q[:3]).as_matrix().tolist(),'ground_to_camera_translation':q[3:6].tolist()}

def apply(folder,bundle):
    folder=Path(folder);meta=json.loads((folder/'result/mesh_meta.json').read_text());return apply_geometry(folder,fit(bundle,meta),bundle)

def apply_geometry(folder,geometry,bundle=None,result=None):
    folder=Path(folder);out=Path(result) if result else folder/'result';meta=json.loads((out/'mesh_meta.json').read_text())
    if geometry.get('video_sha256')!=meta['video_sha256']:raise ValueError('镜面参数属于其他视频')
    evidence=json.loads((out/'person_candidates.json').read_text())
    if evidence['video_sha256']!=meta['video_sha256']:raise ValueError('人物遮罩候选不属于当前视频')
    record=json.loads((folder/'record.json').read_text());archive=folder/'attempts'/f"{record['attempt']:04d}"/'work/reconstruction.npz'
    with np.load(archive,allow_pickle=False) as d:
        vertices=d['vertices'].copy();sam2=d['masks_mirror_sam2'].copy() if 'masks_mirror_sam2' in d else None
    if sam2 is not None and (sam2.ndim!=3 or len(sam2)!=meta['frames'] or sam2.dtype!=np.uint8):raise ValueError('镜中 SAM2 数据无效')
    atlas=cv2.imread(str(out/'person_masks_sam2.png'));cols,rows=meta['mask_atlas_grid'];tw=atlas.shape[1]//cols;th=atlas.shape[0]//rows;atlas[:,:,1]=0;n=np.array(geometry['normal_camera']);distance=geometry['distance_camera_m'];stats=json.loads((out/'person_masks_sam2_stats.json').read_text());accepted=[];size=np.array(meta['image_size'])
    def iou(a,b):
        inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1)
    for i,frame in enumerate(evidence['frames']):
        x=vertices[i]+meta['source_roots'][i];ref=x-2*(x@n-distance)[:,None]*n
        mask=np.zeros((int(size[1]),int(size[0])),np.uint8);valid=False
        if np.all(ref[:,2]>.1):
            p=ref[:,:2]/ref[:,2:]*meta['focal'][i]+size/2;box=np.r_[p.min(0),p.max(0)];real=x[:,:2]/x[:,2:]*meta['focal'][i]+size/2;realbox=np.r_[real.min(0),real.max(0)]
            if sam2 is not None:
                ys,xs=np.where(sam2[i]>140)
                if len(xs):
                    detected=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1])*np.tile(size/np.array(sam2.shape[2:0:-1]),2)
                    if iou(box,detected)>.15 and iou(realbox,detected)<.25:mask=sam2[i];valid=True
            else:
                options=[(iou(box,np.array(c['box'])),c) for c in frame['persons'] if iou(realbox,np.array(c['box']))<.25]
                if options:
                    score,c=max(options,key=lambda a:a[0])
                    if score>.15:cv2.fillPoly(mask,[np.asarray(c['polygon'],np.int32)],255);valid=True
        tile=cv2.resize(mask,(tw,th),interpolation=cv2.INTER_AREA);atlas[i//cols*th:(i//cols+1)*th,i%cols*tw:(i%cols+1)*tw,1]=tile;stats[i]['mirror']=float((tile>140).mean());accepted.append({'accepted':valid,'source':'SAM2_video_propagation' if sam2 is not None else 'paired_ground_mask_association'})
    if not any(row['accepted'] for row in accepted):raise ValueError('镜面拟合通过，但未匹配到镜中人物遮罩；请复核配对点和实测尺寸')
    path=out/'person_masks_sam2.pending.png'
    if not cv2.imwrite(str(path),atlas):raise ValueError('遮罩保存失败')
    geometry['mask_frames']=sum(r['accepted'] for r in accepted)
    path.replace(out/'person_masks_sam2.png');write(out/'mirror_geometry.json',geometry);write(out/'mirror_geometry_frames.json',accepted);write(out/'person_masks_sam2_stats.json',stats);meta['mirror_available']=True;meta['multiview_refined_available']=False;write(out/'mesh_meta.json',meta)
    if bundle:write(out/'paired_ground_calibration.json',bundle);write(out/'ground_calibration.json',bundle['ground']);write(out/'mirror_ground_grid.json',bundle['mirror'])
    quality=out/'quality_report.json'
    if quality.exists():
        report=json.loads(quality.read_text());report.update(mirror_available=True,mirror_mask_frames=geometry['mask_frames'],mask_method=meta.get('mask_method','YOLO_polygon'));write(quality,report)
    write(out/'mirror_calibration_report.json',geometry);return geometry
