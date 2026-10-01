"""Gate independent mirror anatomy before applying small, wrist-anchored hand priors.

Monocular depth offsets are reported, never transferred into primary camera roots.
The hand mesh is a bounded display preview, not MHR parameter refitting.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from run_records import write,sha256

CORE=[5,6,9,10,11,12,13,14]

def reflect(points,normal,distance):
    n=np.asarray(normal,float);n/=np.linalg.norm(n)
    p=np.asarray(points,float)
    return p-2*(p@n-distance)[...,None]*n

def constrain(primary,virtual,valid,normal,distance,fps):
    back=reflect(virtual,normal,distance)
    offset=np.median(primary[:,CORE]-back[:,CORE],axis=1)
    body_error=np.sqrt(np.mean(np.sum((back[:,CORE]+offset[:,None]-primary[:,CORE])**2,axis=-1),axis=1))
    # Camera-space reflected finger offsets share anatomical IDs. Keep the real wrist.
    delta=(back[:,21:41]-back[:,41:42])-(primary[:,21:41]-primary[:,41:42])
    hand_error=np.sqrt(np.mean(np.sum(delta**2,axis=-1),axis=1))
    accepted=np.asarray(valid,bool)&(body_error<.15)&(hand_error<.06)&np.isfinite(back).all(axis=(1,2))
    weights=accepted.astype(float)*.2
    # Fade at evidence boundaries without bleeding corrections into rejected frames.
    weights*=gaussian_filter1d(accepted.astype(float),max(.5,fps*.04))
    correction=delta*weights[:,None,None]
    correction*=np.minimum(1,.008/np.maximum(np.linalg.norm(correction,axis=-1,keepdims=True),1e-12))
    fused=primary.copy();fused[:,21:41]+=correction
    return fused,accepted,body_error,hand_error,offset

def run(folder,result=None):
    folder=Path(folder);out=Path(result) if result else folder/'result'
    meta=json.loads((out/'mesh_meta.json').read_text())
    attempt=json.loads((folder/'record.json').read_text())['attempt']
    archive=folder/'attempts'/f'{attempt:04d}'/'work/reconstruction.npz'
    if not meta.get('multiview_inference_available') or not meta.get('mirror_available'):return None
    provenance=json.loads((out/'multiview_manifest.json').read_text())
    geometry=json.loads((out/'mirror_geometry.json').read_text())
    if provenance['source_sha256']!=meta['video_sha256'] or geometry['video_sha256']!=meta['video_sha256']:raise ValueError('多视角约束与视频不一致')
    with np.load(archive,allow_pickle=False) as d:
        joints=d['joints'].copy();roots=d['source_roots'].copy();vertices=d['vertices'].copy()
        virtual=d['mirror_joints']+d['mirror_roots'][:,None];valid=d['mirror_valid'].copy()
        if len(virtual)!=meta['frames'] or not np.isfinite(virtual).all():raise ValueError('镜中 3D 时间轴无效')
        if not np.allclose(d['mirror_focal'][valid],d['focal'][valid],rtol=.001):raise ValueError('两侧焦距不一致，暂不融合关节')
    fused,accepted,body_error,hand_error,offset=constrain(joints+roots[:,None],virtual,valid,geometry['normal_camera'],geometry['distance_camera_m'],meta['fps'])
    local=(fused-roots[:,None]).astype(np.float32)
    # Preserve camera roots and torso. Body agreement is a gate, not a full-body refit.
    target=out/'multiview_constraints.npz'
    np.savez_compressed(target,joints=local,accepted=accepted,video_sha256=np.array(meta['video_sha256']),geometry_sha256=np.array(sha256(out/'mirror_geometry.json')))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sam3d/joint_fit'))
    from build_full_viewer import deform_hand_vertices
    stable=np.fromfile(out/'mesh_smooth.bin',dtype='<f4').reshape(vertices.shape)
    for i in np.where(accepted)[0]:stable[i]=deform_hand_vertices(stable[i],joints[i,21:42],local[i,21:42])
    preview=out/'mesh_refined.pending.bin';stable.astype('<f4').tofile(preview);preview.replace(out/'mesh_refined.bin')
    report={'video_sha256':meta['video_sha256'],'geometry_sha256':sha256(out/'mirror_geometry.json'),'archive_sha256':sha256(archive),'constraints_sha256':sha256(target),'frames':meta['frames'],'mirror_inferred':int(valid.sum()),'hand_constraints_accepted':int(accepted.sum()),'hand_weight_max':.2,'max_hand_correction_mm':float(np.linalg.norm(local-joints,axis=-1).max()*1000),'core_relative_error_median_m':float(np.median(body_error[valid])),'hand_relative_error_median_m':float(np.median(hand_error[valid])),'global_depth_offset_median_m':float(np.median(np.linalg.norm(offset[valid],axis=-1))),'primary_roots_preserved':True,'body_role':'torso consistency gate; primary full-body geometry preserved','hand_role':'reflected local fingers; primary wrist retained; bounded display deformation','plane_role':'existing heldout-validated 2D mirror fit retained','frames_quality':[{'frame':i,'accepted':bool(accepted[i]),'body_rms_m':float(body_error[i]),'hand_rms_m':float(hand_error[i])} for i in range(len(valid))]}
    write(out/'multiview_constraints_report.json',report)
    meta.update(multiview_refined_available=bool(accepted.any()),multiview_hand_frames=int(accepted.sum()))
    write(out/'mesh_meta.json',meta)
    print(json.dumps({k:v for k,v in report.items() if k!='frames_quality'}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--result',type=Path);a=p.parse_args();run(a.dataset,a.result)
