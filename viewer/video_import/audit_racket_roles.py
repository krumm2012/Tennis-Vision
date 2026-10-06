"""Read-only role and mirror-ray consistency audit; estimated geometry is not truth."""
import argparse,json
from pathlib import Path
import numpy as np
from run_records import sha256,write


def project(p,focal,size):
    return p[:2]/p[2]*focal+np.asarray(size)/2


def ray_pair(real_uv,mirror_uv,focal,size,normal,distance):
    n=np.asarray(normal,float);n=n/np.linalg.norm(n)
    h=np.eye(3)-2*np.outer(n,n);origin=2*distance*n
    a=np.r_[(np.asarray(real_uv)-np.asarray(size)/2)/focal,1.]
    v=np.r_[(np.asarray(mirror_uv)-np.asarray(size)/2)/focal,1.]
    b=h@v
    cross=np.array([[0,-origin[2],origin[1]],[origin[2],0,-origin[0]],[-origin[1],origin[0],0]])
    e=cross@h;lr=e@v;lm=e.T@a;constraint=float(a@lr)
    error=abs(constraint)*focal/max(np.linalg.norm(lr[:2]),1e-12)
    mirror_error=abs(constraint)*focal/max(np.linalg.norm(lm[:2]),1e-12)
    ua=a/np.linalg.norm(a);ub=b/np.linalg.norm(b)
    depth=np.linalg.lstsq(np.column_stack([ua,-ub]),origin,rcond=None)[0]
    gap=np.linalg.norm(depth[0]*ua-(origin+depth[1]*ub))
    return {'real_epipolar_error_px':float(error),'mirror_epipolar_error_px':float(mirror_error),
            'ray_gap_m':float(gap),'positive_depths':bool(np.all(depth>0)),
            'ray_angle_deg':float(np.degrees(np.arccos(np.clip(abs(ua@ub),0,1))))}


def audit(staged,native,observations,output):
    if output.exists():raise ValueError('Use a fresh audit directory')
    meta=json.loads((staged/'mesh_meta.json').read_text());mirror=json.loads((staged/'mirror_geometry.json').read_text())
    marks=json.loads(observations.read_text());kp=json.loads((staged/'racket_keypoints.json').read_text())
    if kp['video_sha256']!=meta['video_sha256'] or mirror['video_sha256']!=meta['video_sha256']:raise ValueError('Source identity mismatch')
    with np.load(native,allow_pickle=False) as d:
        if str(d['video_sha256'])!=meta['video_sha256']:raise ValueError('Native identity mismatch')
        real=d['joints'][:,41]+d['source_roots'];virtual=d['mirror_joints'][:,41]+d['mirror_roots'];valid=d['mirror_valid'];mirror_focal=d['mirror_focal'].copy()
    normal=np.array(mirror['normal_camera']);distance=mirror['distance_camera_m'];scale=meta['image_size'][0]/1280
    rows=[]
    for row in marks['frames']:
        i=row['frame'];focal=meta['focal'][i];wrist=project(real[i],focal,meta['image_size'])
        reflection=real[i]-2*(real[i]@normal-distance)*normal
        mw=project(reflection,focal,meta['image_size']);independent=project(virtual[i],mirror_focal[i],meta['image_size']) if valid[i] else None
        views={}
        for view,own,other in [('points',wrist,mw),('mirror_points',mw,wrist)]:
            points=row.get(view,{})
            if not points:continue
            center=np.array(points.get('head_center',points.get('tip',next(iter(points.values())))))
            own_dist=np.linalg.norm(center-own)/scale;other_dist=np.linalg.norm(center-other)/scale
            views[view]={'own_wrist_distance_canonical_px':float(own_dist),'other_wrist_distance_canonical_px':float(other_dist),
                         'closer_to_other_role':bool(np.linalg.norm(own-other)/scale>50 and own_dist>1.5*other_dist)}
        pairs={}
        for name in set(row.get('points',{}))&set(row.get('mirror_points',{})):
            if name.startswith('rim_'):continue # unconfirmed physical side correspondence
            values=ray_pair(row['points'][name],row['mirror_points'][name],focal,meta['image_size'],normal,distance)
            for key in ['real_epipolar_error_px','mirror_epipolar_error_px']:values[key.replace('_px','_canonical_px')]=values.pop(key)/scale
            pairs[name]=values
        a=kp['frames'][i].get('real',{}).get('points',{});b=kp['frames'][i].get('mirror',{}).get('points',{})
        duplicates=bool(a and b and a==b)
        rows.append({'frame':i,'source':row['source'],'views':views,'ray_pairs':pairs,
                     'independent_vs_plane_mirror_wrist_canonical_px':float(np.linalg.norm(independent-mw)/scale) if independent is not None else None,
                     'same_real_mirror_keypoint_set':duplicates})
    summary={}
    for source in sorted(set(r['source'] for r in rows)):
        selected=[r for r in rows if r['source']==source];errors=[max(p['real_epipolar_error_canonical_px'],p['mirror_epipolar_error_canonical_px']) for r in selected for p in r['ray_pairs'].values()]
        summary[source]={'frames':len(selected),'paired_landmarks':len(errors),'epipolar_median_canonical_px':float(np.median(errors)) if errors else None,
                         'epipolar_p95_canonical_px':float(np.percentile(errors,95)) if errors else None,
                         'closer_to_other_role_frames':[r['frame'] for r in selected if any(v['closer_to_other_role'] for v in r['views'].values())]}
    output.mkdir(parents=True)
    result={'status':'diagnostic_not_accuracy_validation','video_sha256':meta['video_sha256'],'summary':summary,'records':rows,
            'input_sha256':{str(p):sha256(p) for p in [native,observations,staged/'mesh_meta.json',staged/'mirror_geometry.json',staged/'racket_keypoints.json']},
            'limits':'Plane/camera and independent MHR wrists are estimated. Epipolar errors cannot distinguish plane error from point mismatch. Unconfirmed rim sides excluded. No observation, threshold or fit altered.'}
    write(output/'role_geometry_report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['staged','native','observations','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.staged,a.native,a.observations,a.output)['summary']))
