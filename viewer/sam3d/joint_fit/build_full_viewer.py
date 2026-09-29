"""Package native body and joint-fit racket with an approximate hand mesh preview."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np


def deform_hand_vertices(vertices, source_joints, target_joints):
    """Blend joint displacements around source finger bones; retain the original topology.

    This is a visual preview, not MHR pose-parameter skinning or a contact solution.
    All coordinates must use the same body-local frame.
    """
    vertices=np.asarray(vertices)
    source=np.asarray(source_joints)
    delta=np.asarray(target_joints)-source
    if not np.any(delta):return vertices.copy()
    total=np.zeros_like(vertices,dtype=np.float64)
    weight_sum=np.zeros(len(vertices),dtype=np.float64)
    nearest_sq=np.full(len(vertices),np.inf)
    for tip in (0,4,8,12,16):
        for a,b in ((20,tip+3),(tip+3,tip+2),(tip+2,tip+1),(tip+1,tip)):
            bone=source[b]-source[a]
            length_sq=float(bone@bone)
            if length_sq<1e-10:continue
            t=np.clip((vertices-source[a])@bone/length_sq,0,1)
            distance_sq=np.sum((vertices-source[a]-t[:,None]*bone)**2,axis=1)
            nearest_sq=np.minimum(nearest_sq,distance_sq)
            weight=np.exp(-distance_sq/(2*.007**2))
            total+=weight[:,None]*(delta[a]+t[:,None]*(delta[b]-delta[a]))
            weight_sum+=weight
    displacement=total/np.maximum(weight_sum[:,None],1e-30)
    # Full influence close to a bone; fade out before the forearm or torso.
    fade=np.clip((.028-np.sqrt(nearest_sq))/.020,0,1)
    fade=fade*fade*(3-2*fade)
    return vertices+displacement*fade[:,None]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=Path('output/sam3d_cloud'));args=ap.parse_args();root=args.data.resolve();out=root/'joint_fit_v4/full';out.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'scripts'));from install_sam3d_viewer import install
    install(out)
    read=lambda n:json.loads((root/n).read_text());d=read('joint_fit_v4/viewer_data.json');native=np.load(root/'native_hand_sequence/sequence.npz');base=read('temporal_pose.json');meta=read('mesh_meta.json');old=read('racket_poses_v2.json');idx=np.array([r['native_frame'] for r in d['frames']]);j=native['joints'][idx];centers=j[:,[9,10]].mean(1);local=j-centers[:,None,:];camera_roots=centers+native['translations'][idx]
    vertices=(native['vertices'][idx]-centers[:,None,:]).astype('<f4')
    for i,row in enumerate(d['frames']):
        original=j[i,21:42]-centers[i]
        fitted=np.asarray(row['joints_camera'][21:42])-camera_roots[i]
        vertices[i]=deform_hand_vertices(vertices[i],original,fitted)
    vertices.tofile(out/'mesh_local.bin')
    for name in ['mesh_smooth.bin','mesh_refined.bin','mesh_temporal.bin']:
        p=out/name
        if p.is_symlink():p.unlink()
        if p.exists():raise ValueError(f'Refuse to replace unexpected file {p}')
        p.symlink_to('mesh_local.bin')
    for name in ['video.mp4','mesh_faces.bin','wilson_mesh.bin','person_masks_sam2.png','person_masks_sam2_stats.json','mirror_geometry.json','mirror_geometry_frames.json','ground_calibration.json']:
        p=out/name
        if p.is_symlink():p.unlink()
        if p.exists():raise ValueError(f'Refuse to replace unexpected asset {p}')
        p.symlink_to(root/name)
    # New topology correspondence is retained, but old projected texture caches are not.
    np.zeros((len(idx),meta['vertices'],6),dtype='u1').tofile(out/'temporal_texture_sam2.bin')
    meta={**meta,'frames':len(idx),'source_roots':camera_roots.tolist(),'focal':(native['focals'][idx]/2).tolist(),'dataset':'native_body_joint_racket_candidate','hand_mesh_updated':True,'hand_mesh_method':'distance_weighted_joint_preview','hand_mesh_contact_validated':False};(out/'mesh_meta.json').write_text(json.dumps(meta))
    rows=[];rackets=[]
    for i,row in enumerate(d['frames']):
        f=dict(base['frames'][i]);f.update({key:local[i].tolist() for key in ['raw','smooth','refined','temporal']});f['image']=(native['joints2d'][idx[i]]/2).tolist();f['time']=row['time'];rows.append(f)
        R=np.array(row['rotation_camera_columns']);g=np.array(row['grip_camera_m']);t=g-R@np.array([0,d['model']['grip_y_m'],0]);rackets.append({'frame':i,'status':'fitted','source':'joint_sequence_candidate','quality':'joint_candidate','ambiguous':True,'rotation_camera_columns':R.tolist(),'translation_camera_m':t.tolist(),'projected_head_outline':row['ring_image'],'joint_hand_camera':row['joints_camera'][21:42],'joint_hand_image':row['hand_image'],'mask_fit_rms_px':row['metrics']['contour_rms_px'],'wrist_gap_m':float(np.linalg.norm(g-row['joints_camera'][41])),'review_reasons':row['review_reasons'],'contact_proxy_mm':row['metrics']['contact_proxy_mm'],'hand_mesh_updated':True,'hand_mesh_method':'distance_weighted_joint_preview'})
    base['frames']=rows;base['metrics']['acceleration_reduction_percent']=0;base['candidate_no_body_smoothing']=True;(out/'temporal_pose.json').write_text(json.dumps(base,separators=(',',':')))
    pose={**old,'model':d['model'],'method':'joint_sequence_candidate','frames':rackets,'summary':{'frames':len(idx),'silhouette_fitted':0,'constrained_estimate':len(idx)}};pose.pop('grasp_calibration',None);(out/'racket_poses.json').write_text(json.dumps(pose,separators=(',',':')))
    page=(out/'viewer.html').read_text();page=page.replace('<h1>SAM 3D Body · 人体动作展示</h1>','<h1>原片人体 + 联合球拍 · 候选版</h1><p style="border-left:3px solid #ffbf75;padding:12px">手指网格现在跟随优化关节近似变形，橙色线显示目标骨架。尚未验证网格与拍柄接触或标准握法。<a href="../../viewer.html">返回原版 Viewer</a> · <a href="../viewer.html">打开诊断页</a></p>')
    page=page.replace('二阶差分 RMS 降低 ${d.metrics.acceleration_reduction_percent.toFixed(1)}%（平滑程度指标，不是准确率）；实测定位按最低脚估计接地点；腾空时可能错误贴地。','候选模式：原片人体未做额外平滑；球拍已做全片联合优化。手部网格为近似关节位移预览，尚未验证接触。')
    page=page.replace("<script src=\"racket_renderer.js\"></script>",'''<script>window.TENNIS_JOINT_CANDIDATE=true;window.TENNIS_HAND_MESH_PREVIEW=true;$('poseMode').value='raw';$('poseMode').disabled=true;$('bodyMode').value='texture';$('mirrorTexture').checked=false;$('temporalTexture').checked=false;$('mirrorTexture').disabled=true;$('temporalTexture').disabled=true;</script><script src="racket_renderer.js"></script>''')
    (out/'viewer.html').write_text(page);print(out/'viewer.html')
if __name__=='__main__':main()
