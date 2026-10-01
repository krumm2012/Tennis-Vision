"""Reproducible SO(3) jitter / evidence-boundary audit; no causal accuracy claim."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_records import write,sha256


def audit(candidate,observations,output):
    with np.load(candidate,allow_pickle=False) as data:r=data['racket_rotation']
    obs={row['frame']:row for row in json.loads(Path(observations).read_text())['frames']}
    velocity=r[1:]@r[:-1].transpose(0,2,1)
    acceleration=np.degrees(Rotation.from_matrix(velocity[1:]@velocity[:-1].transpose(0,2,1)).magnitude())
    step=np.degrees(Rotation.from_matrix(velocity).magnitude());rows=[]
    for i,value in enumerate(acceleration,1):
        window=[obs.get(k,{}) for k in [i-1,i,i+1]]
        signatures=[(row.get('source','missing'),tuple(sorted(row.get('points',{}))),tuple(sorted(row.get('mirror_points',{})))) for row in window]
        residuals={}
        for view in ['points','mirror_points']:
            for name in ['head_center','tip','handle_end']:
                points=[row.get(view,{}).get(name) for row in window]
                if all(p is not None for p in points):residuals[view+'/'+name]=float(np.linalg.norm(np.array(points[1])-(np.array(points[0])+points[2])/2))
        rows.append({'frame':i,'frame_1based':i+1,'acceleration_deg_frame2':float(value),'evidence_boundary':len(set(signatures))>1,'sources':[row.get('source','missing') for row in window],'midpoint_residual_source_px':residuals})
    top=sorted(rows,key=lambda row:row['acceleration_deg_frame2'],reverse=True)[:15]
    report={'candidate_sha256':sha256(candidate),'observations_sha256':sha256(observations),'step_max_deg':float(step.max()),'acceleration_p95_deg_frame2':float(np.percentile(acceleration,95)),
            'adjacent_normal_reversals':(np.where((r[1:,:,2]*r[:-1,:,2]).sum(1)<0)[0]+1).tolist(),'top15_evidence_boundaries':sum(row['evidence_boundary'] for row in top),'all_evidence_boundaries':sum(row['evidence_boundary'] for row in rows),'evaluated_centers':len(rows),'top15':top,
            'limits':'Boundary association is not proof of causality. No adjacent normal reversal does not establish physical face identity. Pixel residuals are in original image coordinates.'}
    write(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.candidate,a.observations,a.output);print(json.dumps({k:v for k,v in r.items() if k!='top15'}))
