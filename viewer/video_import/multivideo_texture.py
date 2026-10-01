"""Evidence-only MHR UV appearance fusion across approved same-outfit clips.

Each face selects a sharp, non-grazing, depth-visible SAM2-masked observation.
The atlas samples original-resolution video pixels at barycentric surface points.
Unobserved texels stay transparent. No invented back views or geometric merging.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from numba import njit
from run_records import sha256,write,now

def load_data(path):
    # NPZ array access decompresses each time: materialize only the needed fields.
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k] for k in ['vertices','source_roots','masks','masks_mirror_sam2','faces'] if k in archive}

@njit(cache=True)
def uv_lookup(uv,faces,size):
    ids=np.full((size,size),-1,np.int32);bary=np.zeros((size,size,3),np.float32)
    for i in range(len(faces)):
        a,b,c=uv[faces[i]];ax=a[0]*(size-1);ay=(1-a[1])*(size-1);bx=b[0]*(size-1);by=(1-b[1])*(size-1);cx=c[0]*(size-1);cy=(1-c[1])*(size-1)
        den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-8:continue
        for y in range(max(0,int(min(ay,by,cy))),min(size,int(max(ay,by,cy))+2)):
            for x in range(max(0,int(min(ax,bx,cx))),min(size,int(max(ax,bx,cx))+2)):
                aa=((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/den;bb=((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/den;cc=1-aa-bb
                if min(aa,bb,cc)>=-1e-5:ids[y,x]=i;bary[y,x,0]=aa;bary[y,x,1]=bb;bary[y,x,2]=cc
    return ids,bary

@njit(cache=True)
def depth_map(uv,z,faces,width,height):
    depth=np.full((height,width),1e6,np.float32)
    for face in faces:
        a,b,c=face;ax,ay=uv[a];bx,by=uv[b];cx,cy=uv[c]
        if min(z[a],z[b],z[c])<=.1:continue
        den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-8:continue
        for y in range(max(0,int(min(ay,by,cy))),min(height,int(max(ay,by,cy))+2)):
            for x in range(max(0,int(min(ax,bx,cx))),min(width,int(max(ax,bx,cx))+2)):
                aa=((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/den;bb=((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/den;cc=1-aa-bb
                if min(aa,bb,cc)<0:continue
                zz=1/(aa/z[a]+bb/z[b]+cc/z[c])
                if zz<depth[y,x]:depth[y,x]=zz
    return depth

def project(p,focal,size):return p[...,:2]/np.maximum(p[...,2:],.1)*focal+np.asarray(size)/2

def face_neighbors(faces):
    neighbors=np.full((len(faces),3),-1,np.int32);edges={}
    for i,face in enumerate(faces):
        for k in range(3):
            edge=tuple(sorted([int(face[k]),int(face[(k+1)%3])]))
            if edge in edges:
                j,l=edges[edge];neighbors[i,k]=j;neighbors[j,l]=i
            else:edges[edge]=(i,k)
    return neighbors

def coherent_labels(scores,faces,iterations=5):
    """Prefer neighboring faces from a shared view without accepting unseen faces."""
    maximum=scores.max(0);labels=scores.argmax(0);ids=np.arange(len(faces));neighbors=face_neighbors(faces);safe=np.maximum(neighbors,0)
    groups=np.full(len(faces),-1,np.int8)
    for i in ids:
        used=set(groups[neighbors[i][neighbors[i]>=0]])
        groups[i]=next(k for k in range(4) if k not in used)
    for _ in range(iterations):
        for group in range(4):
            candidates=np.concatenate([labels[None],labels[safe].T],axis=0)
            energy=-np.log(np.maximum(scores[candidates,ids]/np.maximum(maximum,1e-8),1e-8))
            for k in range(3):energy+=.25*((candidates!=labels[safe[:,k]])&(neighbors[:,k]>=0)&(maximum[safe[:,k]]>0))
            take=(groups==group)&(maximum>0);labels[take]=candidates[energy.argmin(0),ids][take]
    labels[maximum==0]=0
    return labels

def sample(image,xy):
    if len(xy)>30000:return np.concatenate([sample(image,xy[a:a+30000]) for a in range(0,len(xy),30000)])
    return cv2.remap(image,xy[:,0].astype(np.float32)[:,None],xy[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape((len(xy),)+image.shape[2:])

def views(data,meta,mirror,index):
    camera=data['vertices'][index]+data['source_roots'][index]
    ref=None
    if meta.get('mirror_available') and mirror and 'masks_mirror_sam2' in data:
        n=np.array(mirror['normal_camera']);ref=camera-2*(camera@n-mirror['distance_camera_m'])[:,None]*n
    yield 0,camera,data['masks'][index],ref
    if ref is not None:yield 1,ref,data['masks_mirror_sam2'][index],camera

def visibility(camera,faces,mask,focal,size,other=None):
    size=np.asarray(size);resolution=np.round(size*(1280/size[0])).astype(int)
    uv=project(camera,focal,size);depth=depth_map((uv*resolution/size).astype(np.float32),camera[:,2].astype(np.float32),faces,int(resolution[0]),int(resolution[1]))
    if other is not None:
        other_uv=project(other,focal,size)*resolution/size
        depth=np.minimum(depth,depth_map(other_uv.astype(np.float32),other[:,2].astype(np.float32),faces,int(resolution[0]),int(resolution[1])))
    eroded=cv2.erode(mask,np.ones((3,3),np.uint8));return uv,depth,eroded,resolution

def valid_points(points,mask,depth,focal,size,resolution):
    uv=project(points,focal,size);confidence=sample(mask,uv*np.array(mask.shape[::-1])/size)/255
    measured=sample(depth,uv*resolution/size)
    valid=(points[:,2]>.1)&(uv[:,0]>2)&(uv[:,0]<size[0]-2)&(uv[:,1]>2)&(uv[:,1]<size[1]-2)&(confidence>.92)&(abs(points[:,2]-measured)<.025)
    return valid,uv,confidence

def observe(camera,faces,mask,focal,size,image,other=None):
    uv,depth,mask,res=visibility(camera,faces,mask,focal,size,other);tri=camera[faces];centers=tri.mean(1)
    valid,xy,confidence=valid_points(centers,mask,depth,focal,np.array(size),res)
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    facing=abs((normal*-centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-9)).sum(1))
    corners=uv[faces];a=corners[:,1]-corners[:,0];b=corners[:,2]-corners[:,0];area=abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5
    box=np.r_[xy[valid].min(0),xy[valid].max(0)].astype(int) if valid.any() else np.array([0,0,1,1])
    gray=cv2.cvtColor(image[max(0,box[1]):min(size[1],box[3]+1),max(0,box[0]):min(size[0],box[2]+1)],cv2.COLOR_BGR2GRAY)
    sharpness=float(cv2.Laplacian(gray,cv2.CV_32F).var()) if gray.size else 0
    score=np.where(valid&(facing>.3),confidence*facing*np.sqrt(area)*np.clip(np.sqrt(sharpness/40),.3,2),0).astype(np.float32)
    return score,sample(image,xy),sharpness

def inspect_clip(folder,faces,cache,step=5):
    archive=folder/'attempts/0001/work/reconstruction.npz';meta=json.loads((folder/'result/mesh_meta.json').read_text());video=folder/'source.mp4'
    manifest=json.loads((archive.parent/'remote_manifest.json').read_text())
    if sha256(video)!=meta['video_sha256'] or manifest['source_sha256']!=meta['video_sha256'] or sha256(archive)!=manifest['artifact_sha256']['reconstruction.npz']:raise ValueError('视频/重建来源哈希不一致')
    data=load_data(archive)
    if not np.array_equal(data['faces'],faces):raise ValueError('MHR UV 与重建拓扑不一致')
    geometry=folder/'result/mirror_geometry.json';mirror=json.loads(geometry.read_text()) if geometry.exists() else None
    identity={'archive_sha256':sha256(archive),'video_sha256':meta['video_sha256'],'mirror_sha256':sha256(geometry) if geometry.exists() else None,'step':step,'code_sha256':sha256(Path(__file__))}
    if cache.exists() and cache.with_suffix('.json').exists() and json.loads(cache.with_suffix('.json').read_text())==identity:return
    best=np.zeros(len(faces),np.float32);frames=np.full(len(faces),-1,np.int32);view_ids=np.zeros(len(faces),np.uint8);colors=np.zeros((len(faces),3),np.uint8);heldout=[];all_scores=[];all_colors=[];all_frames=[];all_views=[];cap=cv2.VideoCapture(str(video))
    for index in range(0,meta['frames'],step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
        if not ok:raise ValueError('纹理抽样视频解码失败')
        for view,camera,mask,other in views(data,meta,mirror,index):
            score,rgb,sharpness=observe(camera,faces,mask,meta['focal'][index],meta['image_size'],image,other)
            if index%(step*5)==0:
                ids=np.flatnonzero(score>0);heldout.append((ids,rgb[ids],index,view));continue
            all_scores.append(score);all_colors.append(rgb);all_frames.append(index);all_views.append(view)
            take=score>best;best[take]=score[take];frames[take]=index;view_ids[take]=view;colors[take]=rgb[take]
        if index%50==0:print('Texture observations',folder.name,index,flush=True)
    cap.release()
    hids=np.concatenate([r[0] for r in heldout]);hcolors=np.concatenate([r[1] for r in heldout]);hframes=np.concatenate([np.full(len(r[0]),r[2]) for r in heldout]);hviews=np.concatenate([np.full(len(r[0]),r[3]) for r in heldout])
    np.savez_compressed(cache,score=best,frame=frames,view=view_ids,color=colors,all_score=all_scores,all_color=all_colors,all_frame=all_frames,all_view=all_views,heldout_face=hids,heldout_color=hcolors,heldout_frame=hframes,heldout_view=hviews);write(cache.with_suffix('.json'),identity)

def fuse(batch,layout,output,size=2048,step=5):
    output.mkdir(parents=True,exist_ok=True);manifest=json.loads((batch/'batch_manifest.json').read_text())
    if manifest['status']!='ready':raise ValueError('所有视频生成完成后再进行正式合成')
    folders=[Path(row['folder']) for row in manifest['clips']];rig=np.load(layout,allow_pickle=False);faces=rig['faces'];uv=rig['uv'];uv_faces=rig['uv_faces']
    entries=[];identities=[]
    for folder in folders:
        cache=output/(folder.name+'_observations.npz');inspect_clip(folder,faces,cache,step);entries.append(np.load(cache,allow_pickle=False));identities.append(json.loads(cache.with_suffix('.json').read_text()))
    scores=np.stack([d['score'] for d in entries]);ids=np.arange(len(faces))
    all_scores=np.concatenate([d['all_score'] for d in entries]);label_clip=np.concatenate([np.full(len(d['all_score']),k) for k,d in enumerate(entries)]);label_frame=np.concatenate([d['all_frame'] for d in entries]);label_view=np.concatenate([d['all_view'] for d in entries])
    labels=coherent_labels(all_scores,faces);winner=label_clip[labels];quality=all_scores[labels,ids];frames=label_frame[labels];view_ids=label_view[labels]
    # Channel exposure alignment uses matching surface-face overlap with first clip.
    gains=[];reference=entries[0]['color'].astype(float)
    for d in entries:
        color=d['color'].astype(float);shared=(d['score']>0)&(scores[0]>0)&(reference.min(1)>30)&(color.min(1)>30)
        gain=np.clip(np.median(reference[shared]/color[shared],axis=0),.85,1.18) if shared.sum()>100 else np.ones(3);gains.append(gain)
    face_map,bary=uv_lookup(uv.astype(np.float32),uv_faces.astype(np.int32),size);ys,xs=np.where(face_map>=0);face_ids=face_map[ys,xs];weights=bary[ys,xs]
    rgba=np.zeros((size,size,4),np.uint8);source_clip=np.full((size,size),-1,np.int16);source_frame=np.full((size,size),-1,np.int16);source_view=np.zeros((size,size),np.uint8)
    for clip,folder in enumerate(folders):
        data=load_data(folder/'attempts/0001/work/reconstruction.npz');meta=json.loads((folder/'result/mesh_meta.json').read_text());mirror=json.loads((folder/'result/mirror_geometry.json').read_text());cap=cv2.VideoCapture(str(folder/'source.mp4'))
        pixel_clip=winner[face_ids];pixel_frames=frames[face_ids];pixel_views=view_ids[face_ids]
        for index in np.unique(frames[(winner==clip)&(quality>0)]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,image=cap.read()
            if not ok:raise ValueError('纹理采样帧缺失')
            for view,camera,mask,other in views(data,meta,mirror,int(index)):
                choose=(pixel_clip==clip)&(pixel_frames==index)&(pixel_views==view)&(quality[face_ids]>0);locations=np.flatnonzero(choose)
                if not len(locations):continue
                p=(camera[faces[face_ids[locations]]]*weights[locations,:,None]).sum(1);_,depth,mask,res=visibility(camera,faces,mask,meta['focal'][index],meta['image_size'],other)
                valid,xy,confidence=valid_points(p,mask,depth,meta['focal'][index],np.array(meta['image_size']),res);locations=locations[valid]
                rgb=np.clip(sample(image,xy[valid])*gains[clip],0,255).astype(np.uint8)
                rgba[ys[locations],xs[locations],:3]=rgb;rgba[ys[locations],xs[locations],3]=(confidence[valid]*255).astype(np.uint8)
                source_clip[ys[locations],xs[locations]]=clip;source_frame[ys[locations],xs[locations]]=index;source_view[ys[locations],xs[locations]]=view
        cap.release();print('Atlas sampled',folder.name,flush=True)
    # Gutter RGB prevents dark filtering seams; alpha/coverage evidence stays unchanged.
    known=rgba[:,:,3]>0
    if known.any():
        _,gutter_labels=cv2.distanceTransformWithLabels((~known).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL);rgb=rgba[:,:,:3][known];expanded=rgba[:,:,:3].copy();near=cv2.dilate(known.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~known;expanded[near]=rgb[gutter_labels[near]-1];rgba[:,:,:3]=expanded
    cv2.imwrite(str(output/'body_texture_rgba.png'),rgba)
    np.savez_compressed(output/'texture_sources.npz',clip=source_clip,frame=source_frame,view=source_view,face=face_map)
    rig_rest=rig['rest_vertices'];np.savez_compressed(output/'appearance_mesh.npz',vertices=rig_rest,faces=faces,uv=uv,uv_faces=uv_faces)
    chosen_colors=np.concatenate([d['all_color'].astype(float)*gains[k] for k,d in enumerate(entries)])[labels,ids];validation=[]
    for i,d in enumerate(entries):
        hf=d['heldout_face'];valid=quality[hf]>0;error=abs(chosen_colors[hf[valid]]-d['heldout_color'][valid]*gains[i]).mean(1)
        validation.append({'video':folders[i].name,'train_face_coverage':float((scores[i]>0).mean()),'heldout_samples':int(valid.sum()),'heldout_rgb_mae_median_0_255':float(np.median(error)) if len(error) else None,'gain_bgr':gains[i].tolist()})
    neighbors=face_neighbors(faces);edges=np.c_[np.repeat(ids,3),neighbors.ravel()];edges=edges[(edges[:,1]>=0)&(edges[:,0]<edges[:,1])];supported=(quality[edges[:,0]]>0)&(quality[edges[:,1]]>0)
    transitions=float(np.mean(labels[edges[supported,0]]!=labels[edges[supported,1]]))
    report={'status':'candidate_needs_visual_review','finished_at':now(),'clips':len(folders),'atlas_size':size,'source_video_resolution':'native normalized video, no AI upscaling','layout_sha256':sha256(layout),'code_sha256':sha256(Path(__file__)),'inputs':identities,'single_clip_face_coverage':float((scores[0]>0).mean()),'combined_face_coverage':float((quality>0).mean()),'atlas_observed_texel_fraction':float(known[face_map>=0].mean()),'mirror_texel_fraction':float((source_view[known]==1).mean()) if known.any() else 0,'selection_method':'depth/mask/angle/sharpness with mesh-edge view coherence; no pixel hallucination','neighbor_view_transition_fraction':transitions,'validation':validation,'heldout_rule':f'Every {step*5}th frame excluded from selection; sampled every {step} frames','metric_limits':'appearance residual and template-surface coverage, not calibrated geometry or true novel-view accuracy','unknown_texels':'transparent; gutter RGB does not count as observation','artifact_sha256':{n:sha256(output/n) for n in ['body_texture_rgba.png','texture_sources.npz','appearance_mesh.npz']}}
    write(output/'texture_report.json',report)
    for d in entries:d.close()
    rig.close();return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--layout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,choices=[1024,2048,4096],default=2048);p.add_argument('--step',type=int,default=5);a=p.parse_args()
    if not 1<=a.step<=25:raise ValueError('纹理采样间隔须为 1–25 帧')
    print(json.dumps(fuse(a.batch,a.layout,a.output,a.size,a.step),ensure_ascii=False))
