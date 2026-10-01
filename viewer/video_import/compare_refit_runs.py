"""Compare private joint-fit runs only when their frozen observations match."""
import argparse,json
from pathlib import Path
from run_records import sha256,write


def compare(before,after,output):
    before,after=Path(before),Path(after)
    reports=[json.loads((p/'refit_report.json').read_text()) for p in [before,after]]
    a,b=reports
    for key in ['video_sha256','source_archive_sha256','train_frames','heldout_frames']:
        if a[key]!=b[key]:raise ValueError('Comparison inputs differ: '+key)
    inputs={}
    for name in ['mesh_meta.json','mirror_geometry.json','racket_landmarks.json','racket_keypoints.json','racket_dimensions.json','racket_poses_directional.json']:
        paths=[p.parent/'staged/result'/name for p in [before,after]]
        hashes=[sha256(p) for p in paths]
        if hashes[0]!=hashes[1]:raise ValueError('Frozen evidence differs: '+name)
        inputs[name]=hashes[0]
    for folder,report in zip([before,after],reports):
        if sha256(folder/'mhr_refit_candidate.npz')!=report['candidate_sha256']:raise ValueError('Candidate hash mismatch')
    rows=[]
    def row(name,x,y):
        if x is not None and y is not None:rows.append({'metric':name,'before':x,'after':y,'delta':y-x})
    for section in ['contact_after','heldout_observations_after']:
        for key,stats in a[section].items():
            for stat in ['median','p95']:row(section+'/'+key+'/'+stat,stats[stat],b[section][key][stat])
    row('body_displacement_p95_m',a['body_displacement_p95_m'],b['body_displacement_p95_m'])
    result={'before_report_sha256':sha256(before/'refit_report.json'),'after_report_sha256':sha256(after/'refit_report.json'),
            'input_sha256':inputs,'heldout_frames':a['heldout_frames'],'metrics':rows,
            'before_gate':a['numerical_regression_gate_passed'],'after_gate':b['numerical_regression_gate_passed'],
            'note':'Paired fixed-input comparison. Negative deltas indicate smaller errors, not necessarily meaningful improvement; GPU repeat variance was not measured.'}
    write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['before','after','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=compare(a.before,a.after,a.output);print(json.dumps({'after_gate':r['after_gate'],'metrics':r['metrics']},ensure_ascii=False))
