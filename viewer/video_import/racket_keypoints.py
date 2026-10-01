"""Current-video contour landmarks and mirror-ray shaft evidence.

Unseen butts/throats remain absent. Geometric rim sides do not identify a physical
face. The SAM grip is an association seed, never exported as a measured keypoint.
"""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'viewer/sam3d'))
from fit_racket_pose import project
from fit_racket_video import head_ellipse
from run_records import write,sha256,now

def extract(polygon,grip,confidence):
    points=np.asarray(polygon,float);grip=np.asarray(grip,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<5 or not np.isfinite(points).all():return None
    lo=np.floor(points.min(0)-3).astype(int);hi=np.ceil(points.max(0)+3).astype(int)
    if np.any(hi-lo>350) or np.any(hi-lo<2):return None
    mask=np.zeros(tuple((hi-lo+1)[::-1]),np.uint8);cv2.fillPoly(mask,[np.round(points-lo).astype(np.int32)],1)
    ys,xs=np.where(mask>0)
    if len(xs)<30:return None
    ellipse=head_ellipse(points)
    center=ellipse[0] if ellipse is not None else np.array([xs.mean(),ys.mean()])+lo
    direction=center-grip;length=np.linalg.norm(direction)
    if not 12<length<250:return None
    direction/=length;across=np.array([-direction[1],direction[0]])
    pixels=np.c_[xs,ys]+lo;s=(pixels-grip)@direction;cross=(pixels-grip)@across
    end=s.max();tip_pixels=pixels[s>=end-1.5];tip=tip_pixels.mean(0)
    landmarks={'head_center':center.tolist(),'tip':tip.tolist()};uncertainty={'head_center':2.5 if ellipse is not None else 6.,'tip':3.}
    # A head-only mask cannot supply a butt or throat. Require a visible narrow shaft.
    bins=np.arange(np.floor(s.min()),np.ceil(s.max())+1);width=[]
    for pos in bins:
        q=cross[(s>=pos)&(s<pos+1)];width.append(float(np.ptp(q)+1) if len(q) else 0)
    width=np.array(width);broad=float(width.max());tail=s.min();near=pixels[s<=tail+2].mean(0)
    if tail<-3 and np.linalg.norm(near-grip)<35 and np.median(width[:min(5,len(width))])<max(5,broad*.3):
        landmarks['handle_end']=near.tolist();uncertainty['handle_end']=4.
    neck=np.where((bins>length*.35)&(bins<length*.85)&(width>broad*.35))[0]
    if len(neck):
        j=neck[0];preceding=width[max(0,j-8):j]
        if j>=8 and np.count_nonzero(preceding)>5 and np.median(preceding)<broad*.3:
            pos=bins[j];q=pixels[(s>=pos)&(s<pos+2)]
            landmarks['throat']=q.mean(0).tolist();uncertainty['throat']=5.
    return {'points':landmarks,'uncertainty_px':uncertainty,'confidence':float(confidence),'source':'segmentation_contour_geometry','shaft_direction_image':direction.tolist(),'face_side_identified':False,'head_ellipse_available':ellipse is not None}

def triangulate(real_uv,mirror_uv,focal,size,normal,distance):
    n=np.asarray(normal,float);n/=np.linalg.norm(n);H=np.eye(3)-2*np.outer(n,n)
    real=np.r_[(np.asarray(real_uv)-np.asarray(size)/2)/focal,1.];virtual=np.r_[(np.asarray(mirror_uv)-np.asarray(size)/2)/focal,1.]
    real/=np.linalg.norm(real);reflected=H@virtual;reflected/=np.linalg.norm(reflected);origin=2*distance*n
    depths=np.linalg.lstsq(np.column_stack([real,-reflected]),origin,rcond=None)[0]
    a=depths[0]*real;b=origin+depths[1]*reflected;angle=np.degrees(np.arccos(np.clip(abs(real@reflected),0,1)))
    gap=float(np.linalg.norm(a-b));point=(a+b)/2
    if min(depths)<=0 or point[2]<=.1 or angle<3 or gap>.08:return None
    return point,gap,angle

def role_compatible(center, grip, other, scale):
    if other is None:return True
    own=np.linalg.norm(center-grip);opposite=np.linalg.norm(center-other)
    return not (np.linalg.norm(grip-other)>50*scale and own>opposite*1.5)


def build(folder,result=None):
    folder=Path(folder);out=Path(result) if result else folder/'result';meta=json.loads((out/'mesh_meta.json').read_text());poses=json.loads((out/'racket_poses.json').read_text())
    if poses['video_sha256']!=meta['video_sha256'] or len(poses['frames'])!=meta['frames']:raise ValueError('球拍引导与视频不一致')
    candidates=[]
    for name in ['racket_candidates.json','racket_roi_candidates.json']:
        if (out/name).exists():
            d=json.loads((out/name).read_text())
            if d['video_sha256']!=meta['video_sha256'] or len(d['frames'])!=meta['frames']:raise ValueError('关键点候选属于其他视频')
            candidates.append(d)
    scale=meta['image_size'][0]/1280;size=np.array(meta['image_size']);rows=[]
    geometry=json.loads((out/'mirror_geometry.json').read_text()) if meta.get('mirror_available') else None
    attempt=json.loads((folder/'record.json').read_text())['attempt']
    with np.load(folder/'attempts'/f'{attempt:04d}'/'work/reconstruction.npz',allow_pickle=False) as d:joints=d['joints'].copy();roots=d['source_roots'].copy()
    for i,row in enumerate(poses['frames']):
        target=np.array(row.get('grip_target_camera_m',joints[i,41]+roots[i]));views={}
        real_uv=project(target[None],meta['focal'][i],size)[0]
        mirror_uv=project((target-2*(target@np.array(geometry['normal_camera'])-geometry['distance_camera_m'])*np.array(geometry['normal_camera']))[None],meta['focal'][i],size)[0] if geometry else None
        for view in ['real','mirror']:
            if view=='mirror' and geometry is None:continue
            point=target if view=='real' else target-2*(target@np.array(geometry['normal_camera'])-geometry['distance_camera_m'])*np.array(geometry['normal_camera'])
            uv=project(point[None],meta['focal'][i],size)[0];options=[]
            if view=='real' and row.get('observed_polygon'):
                e=extract(np.array(row['observed_polygon'])/scale,uv/scale,row.get('detection_confidence',.1))
                if e and role_compatible(np.array(e['points']['head_center'])*scale,uv,mirror_uv,scale):options.append((2.,e))
            for source in candidates:
                for c in source['frames'][i]['candidates']:
                    if c.get('view') and c['view']!=view:continue
                    e=extract(np.array(c['polygon'])/scale,uv/scale,c['confidence'])
                    if not e:continue
                    center=np.array(e['points']['head_center'])*scale
                    if not role_compatible(center,uv,mirror_uv if view=='real' else real_uv,scale):continue
                    if view=='mirror' and np.linalg.norm(center-real_uv)<60*scale:continue
                    distance=np.linalg.norm(center-uv)/scale
                    if distance>150:continue
                    options.append((c['confidence']*np.exp(-distance/160),e))
            if options:
                e=max(options,key=lambda q:q[0])[1];e['points']={k:(np.array(p)*scale).tolist() for k,p in e['points'].items()};e['uncertainty_px']={k:v*scale for k,v in e['uncertainty_px'].items()};views[view]=e
        entry={'frame':i,**views}
        if geometry and 'real' in views and 'mirror' in views:
            tri=triangulate(views['real']['points']['head_center'],views['mirror']['points']['head_center'],meta['focal'][i],size,geometry['normal_camera'],geometry['distance_camera_m'])
            if tri:
                head,gap,angle=tri;vector=head-target;length=np.linalg.norm(vector)
                expected=poses['model']['head_center_y_m']-poses['model']['grip_y_m']
                if abs(length-expected)<.08 and gap<.04:
                    entry['stereo_shaft']={'direction_camera':(vector/length).tolist(),'ray_gap_m':gap,'baseline_angle_deg':angle,'grip_to_head_m':float(length),'source':'mirror_ray_head_center_with_primary_grip','physical_face_sign_verified':False}
        rows.append(entry)
    summary={'real_frames':sum('real' in r for r in rows),'mirror_frames':sum('mirror' in r for r in rows),'visible_butt_frames':sum('handle_end' in r.get('real',{}).get('points',{}) for r in rows),'visible_throat_frames':sum('throat' in r.get('real',{}).get('points',{}) for r in rows),'stereo_shaft_frames':sum('stereo_shaft' in r for r in rows)}
    write(out/'racket_keypoints.json',{'video_sha256':meta['video_sha256'],'image_size':meta['image_size'],'fps':meta['fps'],'source_pose_sha256':sha256(out/'racket_poses.json'),'candidate_sha256':{n:sha256(out/n) for n in ['racket_candidates.json','racket_roi_candidates.json'] if (out/n).exists()},'method':'contour_derived_landmarks_with_visibility_gate; not trained keypoint detector','finished_at':now(),'frames':rows,'summary':summary})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);a=p.parse_args();build(a.dataset,a.result)
