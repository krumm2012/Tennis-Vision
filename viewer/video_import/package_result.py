"""Validate a per-video reconstruction before publishing a self-contained Viewer.

NPZ: vertices [F,V,3] in camera axes before source_roots translation,
faces [T,3], source_roots [F,3], focal [F] in normalized video pixels,
masks [F,H,W] uint8 person confidence (0..255). No pickle or legacy-video reuse.
"""
from pathlib import Path
import json, math, shutil, sys
import cv2
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'sam3d/joint_fit'))
from stabilize import smooth_display,stabilize_body
from run_records import sha256, write as write_record

def package(video, archive, destination):
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not 0<fps<=120 or not width or not height:raise ValueError('无法读取标准化视频')
    video_hash = sha256(video)
    provenance = Path(archive).parent / 'multiview_manifest.json'
    multiview_report = None
    for name in ('inference_manifest.json', 'multiview_manifest.json'):
        path = Path(archive).parent / name
        if path.exists():
            report = json.loads(path.read_text())
            if report['source_sha256'] != video_hash or report['frames'] != count:
                raise ValueError('推理来源与当前视频不一致')
            if name == 'multiview_manifest.json':
                if report['image_size'] != [width, height] or report['fps'] != fps:
                    raise ValueError('多视角时间轴与视频不一致')
                multiview_report = report
    with np.load(archive,allow_pickle=False) as data:
        if 'video_sha256' in data and str(data['video_sha256']) != video_hash:
            raise ValueError('重建档案与当前视频 SHA256 不一致')
        vertices=data['vertices'];faces=data['faces'];roots=data['source_roots'];focal=data['focal'];masks=data['masks']
        mirror_masks=data['masks_mirror_sam2'] if 'masks_mirror_sam2' in data else None
        if mirror_masks is not None and (mirror_masks.shape!=masks.shape or mirror_masks.dtype!=np.uint8):raise ValueError('镜中 SAM2 遮罩与真人遮罩不匹配')
        if vertices.ndim!=3 or vertices.shape[0]!=count or vertices.shape[2]!=3 or not 3<=vertices.shape[1]<=100000:raise ValueError('网格数量与视频帧数不一致')
        if not 1<=count<=3000:raise ValueError('视频帧数超出范围')
        if faces.ndim!=2 or faces.shape[1]!=3 or not np.issubdtype(faces.dtype,np.integer) or faces.size==0 or faces.min()<0 or faces.max()>=vertices.shape[1]:raise ValueError('网格索引无效')
        if roots.shape!=(count,3) or focal.shape!=(count,) or np.any(focal<=0):raise ValueError('相机参数无效')
        if masks.ndim!=3 or masks.shape[0]!=count or masks.dtype!=np.uint8 or min(masks.shape[1:])<2:raise ValueError('需要每帧对应的 uint8 人物遮罩')
        if not all(np.isfinite(a).all() for a in (vertices,roots,focal)):raise ValueError('模型输出含无效数值')
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        cols=math.ceil(math.sqrt(count));rows=math.ceil(count/cols);edge=min(512 if mirror_masks is not None else 320,8192//max(cols,rows));scale=edge/max(width,height);tw=max(1,round(width*scale));th=max(1,round(height*scale))
        atlas=np.zeros((rows*th,cols*tw,3),np.uint8);stats=[]
        for i,mask in enumerate(masks):
            tile=cv2.resize(mask,(tw,th),interpolation=cv2.INTER_AREA);atlas[i//cols*th:(i//cols+1)*th,i%cols*tw:(i%cols+1)*tw,2]=tile
            stats.append({'real':float((tile>140).mean()),'mirror':0})
        if not cv2.imwrite(str(destination/'person_masks_sam2.png'),atlas):raise ValueError('无法保存人物遮罩')
        vertices.astype('<f4').tofile(destination/'mesh_local.bin');faces.astype('<u4').tofile(destination/'mesh_faces.bin')
        times=np.arange(count)/fps
        if 'joints' in data and data['joints'].ndim==3 and data['joints'].shape[0]==count and data['joints'].shape[1]>=63:
            stable,_=stabilize_body(vertices,data['joints'],times)
        else:stable=smooth_display(vertices,times)
        stable.tofile(destination/'mesh_smooth.bin')
        for name in ['mesh_refined.bin','mesh_temporal.bin']:(destination/name).hardlink_to(destination/'mesh_local.bin')
        with (destination/'temporal_texture_sam2.bin').open('wb') as stream:stream.truncate(count*vertices.shape[1]*6)
        centers=((vertices.min(axis=1)+vertices.max(axis=1))/2).tolist()
        spans=np.max(vertices.max(axis=1)-vertices.min(axis=1),axis=1).tolist()
        metadata={'video_sha256':video_hash,'frames':count,'vertices':vertices.shape[1],'faces':len(faces),'fps':fps,'image_size':[width,height],'mask_atlas_grid':[cols,rows],'source_roots':roots.tolist(),'focal':focal.tolist(),'display_centers':centers,'display_spans':spans,'mirror_available':False,'stabilization_available':True,'stabilization':{'method':'symmetric_speed_adaptive','max_offset_m':.015},'texture_method':'source_video_projection'}
        if multiview_report is not None:
            report = multiview_report
            shutil.copy2(provenance,destination/provenance.name)
            metadata.update(mask_method=report['sam2_model'],mirror_sam3d_frames=report['mirror_sam3d_frames'],multiview_inference_available=bool(report['mirror_available']))
        def write(name,value):(destination/name).write_text(json.dumps(value,separators=(',',':')))
        write('mesh_meta.json',metadata);write('person_masks_sam2_stats.json',stats);write('mirror_geometry_frames.json',[{'accepted':False} for _ in range(count)]);write('mirror_geometry.json',{'normal_camera':[0,0,1],'distance_camera_m':0})
        camera=vertices+roots[:,None,:]
        valid=camera[:,:,2]>.1
        u=.5+focal[:,None]*camera[:,:,0]/(width*np.maximum(camera[:,:,2],.1))
        v=.5+focal[:,None]*camera[:,:,1]/(height*np.maximum(camera[:,:,2],.1))
        visible=valid&(u>=0)&(u<=1)&(v>=0)&(v<=1)
        write_record(destination/'quality_report.json',{'schema_version':1,'frames':count,'video_sha256':metadata['video_sha256'],'mean_mask_fraction':float(np.mean([s['real'] for s in stats])),'mean_vertices_in_image_fraction':float(visible.mean()),'projection_metric':'vertex projection bounds only; not visible-surface texture coverage','max_display_correction_m':float(np.linalg.norm(stable-vertices,axis=-1).max()),'mirror_available':False,'temporal_texture_available':False})
        shutil.copy2(video,destination/'video.mp4');shutil.copy2(SOURCE/'video_import/dataset.html',destination/'viewer.html')
        shutil.copy2(SOURCE/'video_import/racket_grip_annotation.html',destination/'grip_annotation.html')
