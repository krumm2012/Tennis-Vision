"""User-requested preview of a joint candidate, with reversible file backups.

This does not grant fit acceptance. Source projection geometry, annotations and
UV atlas remain unchanged. Candidate display vertices are rebased into the
existing renderer's source-root coordinates so body and racket stay paired.
"""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from run_records import sha256,write


def publish(result,run):
    result,run=Path(result),Path(run)
    report=json.loads((run/'refit_report.json').read_text());remote=json.loads((run/'remote_manifest.json').read_text())
    candidate=run/'mhr_refit_candidate.npz'
    if sha256(candidate)!=report['candidate_sha256'] or report['candidate_sha256']!=remote['artifact_sha256']['mhr_refit_candidate.npz']:raise ValueError('Candidate hash mismatch')
    meta=json.loads((result/'mesh_meta.json').read_text());staged=run.parent/'staged/result';source=json.loads((staged/'mesh_meta.json').read_text())
    if report['video_sha256']!=meta['video_sha256'] or any(source[k]!=meta[k] for k in ['video_sha256','frames','vertices','fps','image_size']):raise ValueError('Candidate timeline/topology metadata mismatch')
    poses=json.loads((staged/'racket_poses_directional.json').read_text());model=poses['model']
    previous_model=json.loads((result/'racket_poses.json').read_text())['model']
    if any(model[k]!=previous_model[k] for k in ['length_m','head_half_width_m','head_center_y_m','throat_y_m','grip_y_m']):raise ValueError('Candidate racket dimensions differ from the displayed asset')
    if model.get('asset_file','wilson_mesh.bin') not in ['wilson_mesh.bin','wilson_mesh_directional.bin','wilson_mesh_reference.bin']:raise ValueError('Unknown racket asset')
    if not (result/model.get('asset_file','wilson_mesh.bin')).is_file():raise ValueError('Racket asset missing')
    with np.load(candidate,allow_pickle=False) as data:
        vertices=data['vertices'];roots=data['source_roots'];rotations=data['racket_rotation'];translations=data['racket_translation'];joints=data['joints']
        if vertices.shape!=(meta['frames'],meta['vertices'],3) or not all(np.isfinite(a).all() for a in [vertices,roots,rotations,translations,joints]):raise ValueError('Invalid candidate geometry')
        display=(vertices+roots[:,None]-np.asarray(meta['source_roots'])[:,None]).astype('<f4')
        rows=[];grip=np.array([0,model['grip_y_m'],0]);ring=np.array(model['head_outline'])
        objects=np.array([[0,0,0],[0,model['throat_y_m'],0],[0,model['length_m'],0],[model['head_half_width_m'],model['head_center_y_m'],0],[-model['head_half_width_m'],model['head_center_y_m'],0]])
        for i,(r,t) in enumerate(zip(rotations,translations)):
            project=lambda p:(p[:,:2]/p[:,2:]*meta['focal'][i]+np.asarray(meta['image_size'])/2).tolist()
            j=joints[i];axis=j[28]+j[27]-j[40]-j[39];axis/=max(np.linalg.norm(axis),1e-8)
            rows.append({'frame':i,'status':'fitted','quality':'constrained_estimate','source':'mhr_joint_v9_user_preview','rotation_camera_columns':r.tolist(),'translation_camera_m':t.tolist(),'grip_target_camera_m':(r@grip+t).tolist(),'grip_axis_error_deg':float(np.degrees(np.arccos(np.clip(r[:,1]@axis,-1,1)))),'face_angle_to_camera_deg':float(np.degrees(np.arccos(np.clip(abs(r[2,2]),0,1)))),'physical_face_sign_verified':False,'projected_head_outline':project(ring@r.T+t),'projected_points':dict(zip(['handle_end','throat','tip','rim_side','rim_opposite'],project(objects@r.T+t))),'review_reasons':['user_requested_preview_motion_gate_failed']})
    backup=result.parent/'preview_backups'/('before_joint_'+time.strftime('%Y%m%d_%H%M%S'));backup.mkdir(parents=True)
    names=['mesh_refined.bin','mesh_meta.json','racket_poses.json','viewer.html','joint_preview_manifest.json']
    previous={}
    for name in names:
        if (result/name).exists():shutil.copy2(result/name,backup/name);previous[name]=sha256(result/name)
    write(backup/'backup_manifest.json',{'files':previous,'restore_into':str(result.resolve())})
    poses.update(frames=rows,method='mhr_joint_v9_user_preview',summary={'observed':0,'interpolated':len(rows),'hidden':0,'candidate_preview':True})
    meta.update(multiview_refined_available=True,stabilization_available=True,joint_fit_preview={'job_id':remote['remote_job_id'],'accepted':False,'user_requested':True})
    html=Path(__file__).with_name('dataset.html').read_text()
    start=html.index('<select id="pose">');end=html.index('</select>',start)+len('</select>')
    html=html[:start]+'<select id="pose" disabled><option value="refined" id="multiviewPose" selected>联合拟合 v9</option></select>'+html[end:]
    html=html.replace('<h1>新视频 · 三维人体</h1>','<h1>三维人体 · 联合拟合 v9</h1><p role="status" style="color:#ffbd62">候选预览 · 抖动未通过验收 · 原版本已备份</p>')
    # Old directional candidates were fitted to a different body. Do not expose
    # their selector or a local refit action in this paired-body preview.
    html=html.replace('<script src="/assets/dataset_racket_review.js"></script>','')
    pending=result/'mesh_refined.pending.bin';display.tofile(pending);pending.replace(result/'mesh_refined.bin')
    write(result/'racket_poses.json',poses);write(result/'mesh_meta.json',meta)
    manifest={'kind':'user_requested_preview','accepted':False,'job_id':remote['remote_job_id'],'candidate_sha256':report['candidate_sha256'],'backup':str(backup.resolve()),'display_rebase':'candidate_local + candidate_root - existing_source_root','files':{name:sha256(result/name) for name in names[:3]}}
    write(result/'joint_preview_manifest.json',manifest)
    pending=result/'viewer.pending.html';pending.write_text(html);pending.replace(result/'viewer.html')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(json.dumps(publish(a.result,a.run),ensure_ascii=False,indent=2))
