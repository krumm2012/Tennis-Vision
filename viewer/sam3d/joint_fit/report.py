"""Measure all frames, including failed/unknown observations; never accuracy claims."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def project(p,f,principal):return p[...,:2]/p[...,2:]*f[:,None,None]+principal


def metrics(a,R,g,h):
    ring=np.einsum('nij,kj->nki',R,a['ring']-a['model_grip'])+g[:,None,:]
    uv=project(ring,a['focal'],a['principal']);q=np.einsum('nki,nij->nkj',uv-a['ellipse_center'][:,None,:],a['ellipse_axes'])/a['ellipse_radii'][:,None,:]
    er=(np.linalg.norm(q,axis=-1)-1)*np.sqrt(a['ellipse_radii'].prod(-1))[:,None]
    rel=h[:,[0,4,8,12,16]]-g[:,None,:];axial=(rel*R[:,None,:,1]).sum(-1);rad=np.linalg.norm(rel-axial[...,None]*R[:,None,:,1],axis=-1)
    contact=np.abs(rad-.020).mean(1)*1000
    relall=h[:,:20]-g[:,None,:];aall=(relall*R[:,None,:,1]).sum(-1);rall=np.linalg.norm(relall-aall[...,None]*R[:,None,:,1],axis=-1)
    penetration=np.max(np.where((aall>-.045)&(aall<.11),np.maximum(.016-rall,0),0),axis=1)*1000
    jumps=np.r_[0,np.degrees(Rotation.from_matrix(np.einsum('nji,njk->nik',R[:-1],R[1:])).magnitude())]
    return dict(contour_rms_px=np.sqrt(np.mean(er**2,axis=1)),contact_proxy_mm=contact,penetration_proxy_mm=penetration,rotation_step_deg=jumps,ring_image=uv)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('output/sam3d_cloud'));args=ap.parse_args();root=args.data;p=root/'joint_fit_v4';a=dict(np.load(p/'input.npz'));z=np.load(p/'result.npz');quality=json.loads((p/'quality.json').read_text())['frames'];model=json.loads((p/'model.json').read_text());native=np.load(root/'native_hand_sequence/sequence.npz')
    before=metrics(a,a['rotation'],a['grip'],a['hand']);after=metrics(a,z['rotation'],z['grip'],z['hand']);old=json.loads((root/'racket_poses_v2.json').read_text());oldR=np.array([r['rotation_camera_columns'] for r in old['frames']]);oldt=np.array([r['translation_camera_m'] for r in old['frames']]);oldg=np.einsum('nij,j->ni',oldR,a['model_grip'])+oldt;baseline=metrics(a,oldR,oldg,a['hand'])
    observed=a['observation_weight']>0;grasp=a['grasp_weight']>0
    pa=[];ch=[]
    for t in [0,4,8,12,16]:pa.extend([20,t+3,t+2,t+1]);ch.extend([t+3,t+2,t+1,t])
    length0=np.linalg.norm(a['hand'][:,ch]-a['hand'][:,pa],axis=-1);length1=np.linalg.norm(z['hand'][:,ch]-z['hand'][:,pa],axis=-1);bone=np.max(abs(length1-length0),axis=1)*1000
    shift=np.linalg.norm(z['hand']-a['hand'],axis=-1).max(1)*1000;projection=project(z['hand'],a['focal'],a['principal']);image_shift=np.linalg.norm(projection-a['hand_image'],axis=-1).mean(1)
    summary={'frames':len(quality),'manually_selected_frames':0,'hand_mesh_updated':False,'metrics_are_not_ground_truth':True,'observations':int(observed.sum()),'possible_grasp_frames':int(grasp.sum())}
    for name,m in [('initial_native',before),('joint_candidate',after),('v2_same_observations',baseline)]:
        summary[name]={'contour_median_px':float(np.median(m['contour_rms_px'][observed])),'contour_p90_px':float(np.percentile(m['contour_rms_px'][observed],90)),'contact_proxy_median_mm':float(np.median(m['contact_proxy_mm'][grasp])),'rotation_steps_over_45':int((m['rotation_step_deg']>45).sum())}
    summary['v2_same_observations'].pop('contact_proxy_median_mm')
    summary.update(max_bone_length_change_mm=float(bone.max()),max_hand_joint_shift_mm=float(shift.max()),hand_reprojection_mean_px=float(image_shift.mean()))
    rows=[]
    for i,q in enumerate(quality):
        reasons=[]
        if not observed[i]:reasons.append('missing_racket_observation')
        if after['contour_rms_px'][i]>8:reasons.append('contour_mismatch')
        if not grasp[i]:reasons.append('grasp_unknown')
        if grasp[i] and (after['contact_proxy_mm'][i]>8 or after['penetration_proxy_mm'][i]>5):reasons.append('contact_unresolved')
        if bone[i]>3:reasons.append('bone_length_change')
        if shift[i]>20:reasons.append('large_inferred_hand_change')
        if after['rotation_step_deg'][i]>45:reasons.append('rotation_jump')
        if q['mapping_weight']<.3:reasons.append('uncertain_frame_mapping')
        if q['hand_weight']<.25:reasons.append('weak_hand_evidence')
        body=native['joints'][q['native_frame']]+native['translations'][q['native_frame']];body[21:42]=z['hand'][i]
        rows.append({**q,'review_reasons':reasons,'metrics':{k:float(after[k][i]) for k in after if k!='ring_image'},'bone_length_change_mm':float(bone[i]),'hand_joint_shift_mm':float(shift[i]),'hand_image':projection[i].tolist(),'hand_initial_image':a['hand_image'][i].tolist(),'ring_image':after['ring_image'][i].tolist(),'ring_initial_image':before['ring_image'][i].tolist(),'joints_camera':body.tolist(),'hand_initial_camera':a['hand'][i].tolist(),'rotation_camera_columns':z['rotation'][i].tolist(),'grip_camera_m':z['grip'][i].tolist()})
    summary['contact_unresolved_frames']=sum('contact_unresolved' in r['review_reasons'] for r in rows);summary['review_frames']=sum(bool(r['review_reasons']) for r in rows)
    # Conservative acceptance gate: diagnostic only until mesh/rig contact is implemented.
    summary['promotion_status']='diagnostic_only_hand_mesh_not_updated'
    (p/'report.json').write_text(json.dumps(summary,indent=2));(p/'viewer_data.json').write_text(json.dumps({'summary':summary,'model':model,'frames':rows},separators=(',',':')));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
