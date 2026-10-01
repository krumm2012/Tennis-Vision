"""Video-scoped sparse, reviewed racket landmarks; never infer hidden points."""
import numpy as np

PARTS=('handle_end','throat','tip','rim_side','rim_opposite')

def validate(value,meta):
    for key in ['video_sha256','image_size','fps']:
        if value.get(key)!=meta[key]:raise ValueError('球拍校准与视频不一致')
    rows=value.get('frames')
    if not isinstance(rows,list) or len(rows)>meta['frames']:raise ValueError('校准帧列表无效')
    seen=set();result=[]
    for row in rows:
        frame=row.get('frame')
        if type(frame) is not int or not 0<=frame<meta['frames'] or frame in seen:raise ValueError('校准帧号无效或重复')
        seen.add(frame);clean={'frame':frame,'source':'manual_review'}
        for view in ['points','mirror_points']:
            points=row.get(view,{})
            if not isinstance(points,dict) or set(points)-set(PARTS):raise ValueError('未知球拍关键点')
            clean[view]={}
            for name,p in points.items():
                a=np.asarray(p,float)
                if a.shape!=(2,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>meta['image_size']):raise ValueError('球拍关键点必须位于原画面内')
                clean[view][name]=a.tolist()
        if not clean['points'] and not clean['mirror_points']:continue
        confirmed=row.get('face_correspondence_confirmed',False)
        if type(confirmed) is not bool:raise ValueError('拍面确认标记无效')
        label=row.get('side_feature','')
        if not isinstance(label,str) or len(label)>160:raise ValueError('物理侧标记说明无效')
        if confirmed:
            required={'handle_end','tip','rim_side','rim_opposite'}
            if not label.strip() or not required<=set(clean['points']):raise ValueError('确认有向拍面需完整拍柄/拍头/A侧/B侧及固定物理标记说明')
        clean.update(face_correspondence_confirmed=confirmed,side_feature=label.strip());result.append(clean)
    return {'schema_version':1,'video_sha256':meta['video_sha256'],'image_size':meta['image_size'],'fps':meta['fps'],'coordinate_system':'source_video_pixels','frames':sorted(result,key=lambda r:r['frame'])}
