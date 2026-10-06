"""Evidence-only MHR UV appearance fusion across approved same-outfit clips.

Each face selects a sharp, non-grazing, depth-visible SAM2-masked observation.
The atlas samples original-resolution video pixels at barycentric surface points.
Unobserved texels stay transparent. No invented back views or geometric merging.
"""
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np
from numba import njit
from run_records import sha256,write,now

def load_data(path):
    # NPZ array access decompresses each time: materialize only the needed fields.
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k] for k in ['vertices','source_roots','masks','masks_mirror_sam2','faces','mirror_vertices','mirror_roots','mirror_valid','mirror_focal'] if k in archive}

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

def coherent_labels(scores,faces,iterations=5,smoothness=.25):
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
            for k in range(3):energy+=smoothness*((candidates!=labels[safe[:,k]])&(neighbors[:,k]>=0)&(maximum[safe[:,k]]>0))
            take=(groups==group)&(maximum>0);labels[take]=candidates[energy.argmin(0),ids][take]
    labels[maximum==0]=0
    return labels

def sample(image,xy):
    if len(xy)>30000:return np.concatenate([sample(image,xy[a:a+30000]) for a in range(0,len(xy),30000)])
    return cv2.remap(image,xy[:,0].astype(np.float32)[:,None],xy[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape((len(xy),)+image.shape[2:])

def views(data,meta,mirror,index,mirror_source='plane'):
    camera=data['vertices'][index]+data['source_roots'][index]
    ref=None
    if meta.get('mirror_available') and mirror and 'masks_mirror_sam2' in data:
        if mirror_source=='independent':
            if all(k in data for k in ['mirror_vertices','mirror_roots','mirror_valid','mirror_focal']) and data['mirror_valid'][index]:
                if not np.isclose(data['mirror_focal'][index],meta['focal'][index],rtol=0,atol=1e-4):raise ValueError('独立镜中网格焦距与同画面相机不一致')
                # Cloud output already restores x into the original image axes.
                ref=data['mirror_vertices'][index]+data['mirror_roots'][index]
        else:
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

def valid_points(points,mask,depth,focal,size,resolution,depth_tolerance=.025):
    uv=project(points,focal,size);confidence=sample(mask,uv*np.array(mask.shape[::-1])/size)/255
    measured=sample(depth,uv*resolution/size)
    valid=(points[:,2]>.1)&(uv[:,0]>2)&(uv[:,1]>2)&(uv[:,0]<size[0]-2)&(uv[:,1]<size[1]-2)&(confidence>.92)&(abs(points[:,2]-measured)<depth_tolerance)
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

def fallback_candidates(scores,primary,frames,step,limit,face_chunk=256,train_allowed=None):
    """Rank at most limit alternative TRAIN observations per supported face.

    Memory for ranking is bounded by observations * face_chunk, not a full sort.
    Stable argmax ties prefer the earlier frozen observation. Heldout frames are
    rejected independently of cache construction as a second isolation guard.
    """
    if not 1<=limit<=8:raise ValueError('fallback candidates must be 1–8')
    if step<1:raise ValueError('step must be positive')
    result=np.full((limit,scores.shape[1]),-1,np.int32)
    train=(np.asarray(frames)>=0)&(np.asarray(frames)%(step*5)!=0) if train_allowed is None else np.asarray(train_allowed,dtype=bool)
    for start in range(0,scores.shape[1],face_chunk):
        stop=min(start+face_chunk,scores.shape[1]);cols=np.arange(stop-start)
        ranked=np.asarray(scores[:,start:stop],dtype=np.float32).copy()
        ranked[~train]=0;ranked[~np.isfinite(ranked)]=0
        supported=scores[primary[start:stop],np.arange(start,stop)]>0
        ranked[:,~supported]=0;ranked[primary[start:stop],cols]=0
        for rank in range(limit):
            chosen=ranked.argmax(0);good=ranked[chosen,cols]>0
            result[rank,start:stop]=np.where(good,chosen,-1)
            ranked[chosen,cols]=0
    return result

def sample_surface_texels(camera,faces,face_ids,weights,mask,focal,size,image,other,gain,face_depth_tolerance=None):
    """Return actual video pixels only where the surface passes mask/depth tests."""
    _,depth,eroded,res=visibility(camera,faces,mask,focal,size,other)
    valid=np.zeros(len(face_ids),bool);rgb=np.zeros((len(face_ids),3),np.uint8);alpha=np.zeros(len(face_ids),np.uint8)
    for start in range(0,len(face_ids),30000):
        stop=min(start+30000,len(face_ids))
        points=(camera[faces[face_ids[start:stop]]]*weights[start:stop,:,None]).sum(1)
        tolerance=.025 if face_depth_tolerance is None else face_depth_tolerance[face_ids[start:stop]]
        ok,xy,confidence=valid_points(points,eroded,depth,focal,np.asarray(size),res,tolerance)
        valid[start:stop]=ok
        if ok.any():
            rgb[start:stop][ok]=np.clip(sample(image,xy[ok])*gain,0,255).astype(np.uint8)
            alpha[start:stop][ok]=(confidence[ok]*255).astype(np.uint8)
    return valid,rgb,alpha

def fill_pixel_fallback(candidates,face_ids,ys,xs,rgba,source_clip,source_frame,source_view,
                        label_clip,label_frame,label_view,sampler,fallback_rank=None):
    """Fill only missing texels; sampler receives a frozen observation and locations.

    Labels encode TRAIN sources, and all pixels retain the actual clip/frame/view.
    Group one observation at a time, bounding decoded frames and depth-map memory.
    """
    attempts=0
    for rank,row in enumerate(candidates,1):
        labels=row[face_ids]
        missing=rgba[ys,xs,3]==0
        for label in np.unique(labels[missing&(labels>=0)]):
            locations=np.flatnonzero(missing&(labels==label))
            attempts+=len(locations)
            valid,rgb,alpha=sampler(int(label),locations)
            take=locations[valid]
            rgba[ys[take],xs[take],:3]=rgb[valid];rgba[ys[take],xs[take],3]=alpha[valid]
            source_clip[ys[take],xs[take]]=label_clip[label]
            source_frame[ys[take],xs[take]]=label_frame[label]
            source_view[ys[take],xs[take]]=label_view[label]
            if fallback_rank is not None:fallback_rank[ys[take],xs[take]]=rank
    return attempts

def inspect_clip(folder,faces,cache,step=5,mirror_source='plane'):
    archive=folder/'attempts/0001/work/reconstruction.npz';meta=json.loads((folder/'result/mesh_meta.json').read_text());video=folder/'source.mp4'
    manifest=json.loads((archive.parent/'remote_manifest.json').read_text())
    if sha256(video)!=meta['video_sha256'] or manifest['source_sha256']!=meta['video_sha256'] or sha256(archive)!=manifest['artifact_sha256']['reconstruction.npz']:raise ValueError('视频/重建来源哈希不一致')
    data=load_data(archive)
    if not np.array_equal(data['faces'],faces):raise ValueError('MHR UV 与重建拓扑不一致')
    geometry=folder/'result/mirror_geometry.json';mirror=json.loads(geometry.read_text()) if geometry.exists() else None
    identity={'archive_sha256':sha256(archive),'video_sha256':meta['video_sha256'],'mesh_meta_sha256':sha256(folder/'result/mesh_meta.json'),'mirror_sha256':sha256(geometry) if geometry.exists() else None,'step':step,'mirror_source':mirror_source,'code_sha256':sha256(Path(__file__))}
    if cache.exists() and cache.with_suffix('.json').exists():
        saved=json.loads(cache.with_suffix('.json').read_text());cache_hash=saved.pop('observation_cache_sha256',None)
        if saved==identity and cache_hash==sha256(cache):return
    best=np.zeros(len(faces),np.float32);frames=np.full(len(faces),-1,np.int32);view_ids=np.zeros(len(faces),np.uint8);colors=np.zeros((len(faces),3),np.uint8);heldout=[];all_scores=[];all_colors=[];all_frames=[];all_views=[];cap=cv2.VideoCapture(str(video))
    for index in range(0,meta['frames'],step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
        if not ok:raise ValueError('纹理抽样视频解码失败')
        for view,camera,mask,other in views(data,meta,mirror,index,mirror_source):
            score,rgb,sharpness=observe(camera,faces,mask,meta['focal'][index],meta['image_size'],image,other)
            if index%(step*5)==0:
                ids=np.flatnonzero(score>0);heldout.append((ids,rgb[ids],index,view));continue
            all_scores.append(score);all_colors.append(rgb);all_frames.append(index);all_views.append(view)
            take=score>best;best[take]=score[take];frames[take]=index;view_ids[take]=view;colors[take]=rgb[take]
        if index%50==0:print('Texture observations',folder.name,index,flush=True)
    cap.release()
    hids=np.concatenate([r[0] for r in heldout]);hcolors=np.concatenate([r[1] for r in heldout]);hframes=np.concatenate([np.full(len(r[0]),r[2]) for r in heldout]);hviews=np.concatenate([np.full(len(r[0]),r[3]) for r in heldout])
    np.savez_compressed(cache,score=best,frame=frames,view=view_ids,color=colors,all_score=all_scores,all_color=all_colors,all_frame=all_frames,all_view=all_views,heldout_face=hids,heldout_color=hcolors,heldout_frame=hframes,heldout_view=hviews)
    identity['observation_cache_sha256']=sha256(cache);write(cache.with_suffix('.json'),identity)

def validate_fusion_group(group_manifest,batch_clips,layout):
    from appearance_training_manifest import verify_fusion_group
    manifest=json.loads(Path(group_manifest).read_text())
    identity=verify_fusion_group(manifest,batch_clips)
    if manifest['layout']['sha256']!=sha256(layout):raise ValueError('fusion layout differs from group manifest')
    rows=[]
    for entry in batch_clips:
        folder=Path(entry['folder']).resolve()
        row=next(r for r in manifest['clips'] if Path(r['inputs']['video']['path']).resolve()==folder/'source.mp4')
        for key,relative in [('mesh_meta','result/mesh_meta.json'),('native_archive','attempts/0001/work/reconstruction.npz'),('native_manifest','attempts/0001/work/remote_manifest.json')]:
            if row['inputs'][key]['sha256']!=sha256(folder/relative):raise ValueError('fusion asset differs from group manifest: '+key)
        rows.append(row)
    return identity,rows

def validate_cache_split(entry,row):
    train=set(row['frames']['train_frames']);heldout=set(row['frames']['heldout_frames'])
    if not set(np.asarray(entry['all_frame']).tolist())<=train:raise ValueError('observation cache contains non-TRAIN frames')
    if not set(np.asarray(entry['heldout_frame']).tolist())<=heldout:raise ValueError('observation cache heldout differs from group manifest')

def validate_cache_binding(identity,cache,folder,allow_legacy=False):
    missing=[k for k in ('mesh_meta_sha256','observation_cache_sha256') if k not in identity]
    if missing and not allow_legacy:raise ValueError('legacy observation cache requires explicit allow_legacy_cache or fresh observations')
    if 'mesh_meta_sha256' in identity and identity['mesh_meta_sha256']!=sha256(folder/'result/mesh_meta.json'):raise ValueError('cache mesh metadata changed')
    if 'observation_cache_sha256' in identity and identity['observation_cache_sha256']!=sha256(cache):raise ValueError('cache NPZ content changed')
    return {'status':'legacy_unverified_generation_binding' if missing else 'verified','missing_fields':missing}

def fuse(batch,layout,output,size=2048,step=5,mirror_source='plane',selection='quality',observation_root=None,pixel_fallback=False,fallback_max_candidates=3,group_manifest=None,allow_legacy_cache=False,face_depth_tolerance=None,fixed_exposure_gains=None):
    if pixel_fallback and not 1<=fallback_max_candidates<=8:raise ValueError('fallback candidates must be 1–8')
    if pixel_fallback and group_manifest is None:raise ValueError('pixel fallback requires an explicit verified group manifest')
    manifest=json.loads((batch/'batch_manifest.json').read_text())
    if manifest['status']!='ready':raise ValueError('所有视频生成完成后再进行正式合成')
    group_identity,group_rows=validate_fusion_group(group_manifest,manifest['clips'],layout) if group_manifest else (None,None)
    output.mkdir(parents=True,exist_ok=True)
    folders=[Path(row['folder']) for row in manifest['clips']];rig=np.load(layout,allow_pickle=False);faces=rig['faces'];uv=rig['uv'];uv_faces=rig['uv_faces']
    if face_depth_tolerance is not None:
        face_depth_tolerance=np.asarray(face_depth_tolerance,dtype=np.float32)
        if not pixel_fallback or face_depth_tolerance.shape!=(len(faces),) or not np.isfinite(face_depth_tolerance).all() or np.any(face_depth_tolerance<=0):
            raise ValueError('per-face depth tolerances require pixel fallback and one finite positive value per face')
    if fixed_exposure_gains is not None:
        fixed_exposure_gains=np.asarray(fixed_exposure_gains,dtype=float)
        if fixed_exposure_gains.shape!=(len(folders),3) or not np.isfinite(fixed_exposure_gains).all() or np.any(fixed_exposure_gains<=0):
            raise ValueError('fixed exposure gains must be positive BGR values for every clip')
    entries=[];identities=[];cache_bindings=[]
    for folder in folders:
        cache=(observation_root or output)/(folder.name+'_observations.npz')
        if observation_root is None:inspect_clip(folder,faces,cache,step,mirror_source)
        identity=json.loads(cache.with_suffix('.json').read_text())
        cache_bindings.append(validate_cache_binding(identity,cache,folder,allow_legacy=allow_legacy_cache or group_manifest is None))
        if identity['step']!=step or identity['mirror_source']!=mirror_source or identity['video_sha256']!=sha256(folder/'source.mp4') or identity['archive_sha256']!=sha256(folder/'attempts/0001/work/reconstruction.npz'):raise ValueError('固定观测缓存与本次输入不一致')
        geometry=folder/'result/mirror_geometry.json'
        if identity['mirror_sha256']!=(sha256(geometry) if geometry.exists() else None):raise ValueError('固定观测缓存镜面来源已变化')
        entry=np.load(cache,allow_pickle=False)
        if group_rows is not None:validate_cache_split(entry,group_rows[len(entries)])
        entries.append(entry);identities.append(identity)
    scores=np.stack([d['score'] for d in entries]);ids=np.arange(len(faces))
    all_scores=np.concatenate([d['all_score'] for d in entries]);label_clip=np.concatenate([np.full(len(d['all_score']),k) for k,d in enumerate(entries)]);label_frame=np.concatenate([d['all_frame'] for d in entries]);label_view=np.concatenate([d['all_view'] for d in entries])
    # Channel exposure alignment uses matching surface-face overlap with first clip.
    gains=[];reference=entries[0]['color'].astype(float)
    for d in entries:
        color=d['color'].astype(float);shared=(d['score']>0)&(scores[0]>0)&(reference.min(1)>30)&(color.min(1)>30)
        gain=np.clip(np.median(reference[shared]/color[shared],axis=0),.85,1.18) if shared.sum()>100 else np.ones(3);gains.append(gain)
    if fixed_exposure_gains is not None:gains=list(fixed_exposure_gains)
    if selection=='consistency':
        from texture_consistency import consistency_scores
        colors=np.concatenate([d['all_color'].astype(np.float32)*gains[k] for k,d in enumerate(entries)])
        adjusted,_=consistency_scores(all_scores,colors,label_clip)
        ranking_scores=adjusted;labels=coherent_labels(adjusted,faces,smoothness=.6)
    elif selection=='quality':
        ranking_scores=all_scores;labels=coherent_labels(all_scores,faces)
    else:raise ValueError('Unknown texture selection method')
    winner=label_clip[labels];quality=all_scores[labels,ids];frames=label_frame[labels];view_ids=label_view[labels]
    face_map,bary=uv_lookup(uv.astype(np.float32),uv_faces.astype(np.int32),size);ys,xs=np.where(face_map>=0);face_ids=face_map[ys,xs];weights=bary[ys,xs]
    rgba=np.zeros((size,size,4),np.uint8);source_clip=np.full((size,size),-1,np.int16);source_frame=np.full((size,size),-1,np.int32 if pixel_fallback else np.int16);source_view=np.zeros((size,size),np.uint8)
    for clip,folder in enumerate(folders):
        data=load_data(folder/'attempts/0001/work/reconstruction.npz');meta=json.loads((folder/'result/mesh_meta.json').read_text());geometry=folder/'result/mirror_geometry.json';mirror=json.loads(geometry.read_text()) if geometry.exists() else None;cap=cv2.VideoCapture(str(folder/'source.mp4'))
        pixel_clip=winner[face_ids];pixel_frames=frames[face_ids];pixel_views=view_ids[face_ids]
        for index in np.unique(frames[(winner==clip)&(quality>0)]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,image=cap.read()
            if not ok:raise ValueError('纹理采样帧缺失')
            for view,camera,mask,other in views(data,meta,mirror,int(index),mirror_source):
                choose=(pixel_clip==clip)&(pixel_frames==index)&(pixel_views==view)&(quality[face_ids]>0);locations=np.flatnonzero(choose)
                if not len(locations):continue
                if pixel_fallback:
                    valid,rgb,alpha=sample_surface_texels(camera,faces,face_ids[locations],weights[locations],mask,
                        meta['focal'][index],meta['image_size'],image,other,gains[clip],face_depth_tolerance)
                    take=locations[valid]
                    rgba[ys[take],xs[take],:3]=rgb[valid];rgba[ys[take],xs[take],3]=alpha[valid]
                    source_clip[ys[take],xs[take]]=clip;source_frame[ys[take],xs[take]]=index;source_view[ys[take],xs[take]]=view
                    continue
                p=(camera[faces[face_ids[locations]]]*weights[locations,:,None]).sum(1);_,depth,mask,res=visibility(camera,faces,mask,meta['focal'][index],meta['image_size'],other)
                valid,xy,confidence=valid_points(p,mask,depth,meta['focal'][index],np.array(meta['image_size']),res);locations=locations[valid]
                rgb=np.clip(sample(image,xy[valid])*gains[clip],0,255).astype(np.uint8)
                rgba[ys[locations],xs[locations],:3]=rgb;rgba[ys[locations],xs[locations],3]=(confidence[valid]*255).astype(np.uint8)
                source_clip[ys[locations],xs[locations]]=clip;source_frame[ys[locations],xs[locations]]=index;source_view[ys[locations],xs[locations]]=view
        cap.release();print('Atlas sampled',folder.name,flush=True)
    primary_known=rgba[:,:,3]>0;fallback_attempts=0
    fallback_rank=np.where(primary_known,0,-1).astype(np.int8) if pixel_fallback else None
    if pixel_fallback:
        allowed=np.concatenate([np.isin(d['all_frame'],row['frames']['train_frames']) for d,row in zip(entries,group_rows)])
        candidates=fallback_candidates(ranking_scores,labels,label_frame,step,fallback_max_candidates,train_allowed=allowed)
        # One clip reconstruction, video capture, decoded image and visibility map
        # at a time. Reopen on clip transitions; do not cache all training frames.
        active_clip=None;cap=None;data=meta=mirror=None
        def sampler(label,locations):
            nonlocal active_clip,cap,data,meta,mirror
            clip=int(label_clip[label]);index=int(label_frame[label]);view=int(label_view[label])
            if active_clip!=clip:
                if cap is not None:cap.release()
                folder=folders[clip];data=load_data(folder/'attempts/0001/work/reconstruction.npz')
                meta=json.loads((folder/'result/mesh_meta.json').read_text())
                geometry=folder/'result/mirror_geometry.json';mirror=json.loads(geometry.read_text()) if geometry.exists() else None
                cap=cv2.VideoCapture(str(folder/'source.mp4'));active_clip=clip
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
            if not ok:raise ValueError('fallback training frame decode failed')
            for actual_view,camera,mask,other in views(data,meta,mirror,index,mirror_source):
                if actual_view==view:
                    return sample_surface_texels(camera,faces,face_ids[locations],weights[locations],mask,
                        meta['focal'][index],meta['image_size'],image,other,gains[clip],face_depth_tolerance)
            return np.zeros(len(locations),bool),np.zeros((len(locations),3),np.uint8),np.zeros(len(locations),np.uint8)
        try:
            fallback_attempts=fill_pixel_fallback(candidates,face_ids,ys,xs,rgba,source_clip,source_frame,source_view,
                label_clip,label_frame,label_view,sampler,fallback_rank)
        finally:
            if cap is not None:cap.release()
    # Gutter RGB prevents dark filtering seams; alpha/coverage evidence stays unchanged.
    known=rgba[:,:,3]>0
    if known.any():
        _,gutter_labels=cv2.distanceTransformWithLabels((~known).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL);rgb=rgba[:,:,:3][known];expanded=rgba[:,:,:3].copy();near=cv2.dilate(known.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~known;expanded[near]=rgb[gutter_labels[near]-1];rgba[:,:,:3]=expanded
    cv2.imwrite(str(output/'body_texture_rgba.png'),rgba)
    np.savez_compressed(output/'texture_sources.npz',clip=source_clip,frame=source_frame,view=source_view,face=face_map,
        **({'fallback_rank':fallback_rank} if pixel_fallback else {}))
    rig_rest=rig['rest_vertices'];np.savez_compressed(output/'appearance_mesh.npz',vertices=rig_rest,faces=faces,uv=uv,uv_faces=uv_faces)
    chosen_colors=np.concatenate([d['all_color'].astype(float)*gains[k] for k,d in enumerate(entries)])[labels,ids];validation=[]
    for i,d in enumerate(entries):
        hf=d['heldout_face'];valid=quality[hf]>0;error=abs(chosen_colors[hf[valid]]-d['heldout_color'][valid]*gains[i]).mean(1)
        validation.append({'video':folders[i].name,'train_face_coverage':float((scores[i]>0).mean()),'heldout_samples':int(valid.sum()),'heldout_rgb_mae_median_0_255':float(np.median(error)) if len(error) else None,'gain_bgr':gains[i].tolist()})
    neighbors=face_neighbors(faces);edges=np.c_[np.repeat(ids,3),neighbors.ravel()];edges=edges[(edges[:,1]>=0)&(edges[:,0]<edges[:,1])];supported=(quality[edges[:,0]]>0)&(quality[edges[:,1]]>0)
    transitions=float(np.mean(labels[edges[supported,0]]!=labels[edges[supported,1]]))
    report={'status':'candidate_needs_visual_review','finished_at':now(),'clips':len(folders),'atlas_size':size,'source_video_resolution':'native normalized video, no AI upscaling','layout_sha256':sha256(layout),'code_sha256':sha256(Path(__file__)),'inputs':identities,'single_clip_face_coverage':float((scores[0]>0).mean()),'combined_face_coverage':float((quality>0).mean()),'atlas_observed_texel_fraction':float(known[face_map>=0].mean()),'mirror_texel_fraction':float((source_view[known]==1).mean()) if known.any() else 0,'selection_method':'depth/mask/angle/sharpness with mesh-edge view coherence; no pixel hallucination','neighbor_view_transition_fraction':transitions,'validation':validation,'heldout_rule':f'Every {step*5}th frame excluded from selection; sampled every {step} frames','metric_limits':'appearance residual and template-surface coverage, not calibrated geometry or true novel-view accuracy','unknown_texels':'transparent; gutter RGB does not count as observation','artifact_sha256':{n:sha256(output/n) for n in ['body_texture_rgba.png','texture_sources.npz','appearance_mesh.npz']}}
    atlas_texels=int((face_map>=0).sum());primary_count=int(primary_known[face_map>=0].sum())
    fallback_count=int((known&~primary_known)[face_map>=0].sum())
    report['pixel_fallback']={'enabled':bool(pixel_fallback),'max_candidates_per_face':fallback_max_candidates if pixel_fallback else 0,
        'policy':'positive TRAIN face score, descending selection score, stable observation order; skip primary; fill transparent texels only; same mask/depth checks; no heldout or synthesized colors',
        'code_sha256':sha256(Path(__file__)),'ranking_face_chunk':256,'sampling_texel_chunk':30000,'rank_provenance':'0 primary, 1..max_candidates fallback, -1 unobserved' if pixel_fallback else None,
        'metric_limits':'heldout_rgb_mae remains face-center primary selection residual; it does not measure fallback atlas pixel error; remaining transparency means no valid evidence in bounded positive-face-score candidate search, not all sources invisible',
        'primary_observed_texels':primary_count,'fallback_added_texels':fallback_count,'candidate_texel_attempts':fallback_attempts,
        'primary_observed_texel_fraction':primary_count/atlas_texels if atlas_texels else 0,
        'fallback_added_texel_fraction':fallback_count/atlas_texels if atlas_texels else 0}
    report['mirror_source']=mirror_source
    report['appearance_group']={'identity':group_identity,'manifest_sha256':sha256(group_manifest) if group_manifest else None,'verified':group_manifest is not None,'scope':'body material; racket separate; explicit user labels, not automatic identity recognition'}
    report['observation_cache_binding']=cache_bindings
    report['selection']=selection
    report['controlled_sampling']={'per_face_depth_tolerance_m':None if face_depth_tolerance is None else {'min':float(face_depth_tolerance.min()),'max':float(face_depth_tolerance.max()),'sha256':hashlib.sha256(face_depth_tolerance.tobytes()).hexdigest()},'exposure_gains_frozen':fixed_exposure_gains is not None}
    report['observation_cache_sha256']={folder.name:sha256((observation_root or output)/(folder.name+'_observations.npz')) for folder in folders}
    if selection=='consistency':
        report['selection_method']='clip-balanced training color consistency soft penalty + mesh-edge view coherence 0.6; actual single-source pixels, no invented color'
        report['consistency_code_sha256']=sha256(Path(__file__).with_name('texture_consistency.py'))
    write(output/'texture_report.json',report)
    for d in entries:d.close()
    rig.close();return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--layout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,choices=[1024,2048,4096],default=2048);p.add_argument('--step',type=int,default=5);p.add_argument('--mirror-source',choices=['plane','independent'],default='plane');p.add_argument('--allow-legacy-cache',action='store_true');p.add_argument('--group-manifest',type=Path);p.add_argument('--pixel-fallback',action='store_true');p.add_argument('--fallback-max-candidates',type=int,choices=range(1,9),default=3);a=p.parse_args()
    if not 1<=a.step<=25:raise ValueError('纹理采样间隔须为 1–25 帧')
    print(json.dumps(fuse(a.batch,a.layout,a.output,a.size,a.step,a.mirror_source,pixel_fallback=a.pixel_fallback,fallback_max_candidates=a.fallback_max_candidates,group_manifest=a.group_manifest,allow_legacy_cache=a.allow_legacy_cache),ensure_ascii=False))
