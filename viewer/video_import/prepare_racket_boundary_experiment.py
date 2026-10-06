"""Freeze a matched A/B pair and reserve fresh annotation frames before fitting."""
import argparse,copy,json,shutil
from pathlib import Path
from run_records import sha256,write

RESERVED=[30,80,130,155,210,230]
ABLATION=list(range(180,185))


def isolate(keypoints,manual,reserved,ablation=()):
    if set(reserved)&{r['frame'] for r in manual['frames']}:raise ValueError('Reserved frames overlap prior reviewed labels')
    if set(reserved)&set(ablation):raise ValueError('Validation and ablation regions overlap')
    result=copy.deepcopy(keypoints);events=[]
    for i in reserved:
        row=result['frames'][i]
        for view in ['real','mirror']:
            if view in row:events.append({'frame':i,'view':view,'reason':'reserved_for_future_annotation'})
            row.pop(view,None)
        row.pop('stereo_shaft',None)
    for i in ablation:
        if i in {r['frame'] for r in manual['frames']}:raise ValueError('Cannot ablate reviewed frame')
        row=result['frames'][i]
        if 'mirror' in row:events.append({'frame':i,'view':'mirror','reason':'diagnostic_boundary_ablation'})
        row.pop('mirror',None);row.pop('stereo_shaft',None)
    return result,events


def prepare(source,destination):
    if destination.exists():raise ValueError('Use a fresh experiment directory')
    kp=json.loads((source/'racket_keypoints.json').read_text());manual=json.loads((source/'racket_landmarks.json').read_text())
    for name,ablation in [('control',[]),('mirror_ablation',ABLATION)]:
        out=destination/name/'staged/result';out.mkdir(parents=True)
        for path in source.glob('*.json'):shutil.copy2(path,out/path.name)
        altered,events=isolate(kp,manual,RESERVED,ablation);write(out/'racket_keypoints.json',altered)
        review=json.loads((out/'racket_review_frames.json').read_text());review['keypoints_sha256']=sha256(out/'racket_keypoints.json');write(out/'racket_review_frames.json',review)
        write(destination/name/'experiment.json',{'source_sha256':sha256(source/'racket_keypoints.json'),'reserved_frames':RESERVED,'ablation_frames':ablation,'events':events,'status':'diagnostic_only_not_acceptance'})
    write(destination/'protocol.json',{'reserved_frames':RESERVED,'ablation_frames':ABLATION,'video_sha256':kp['video_sha256'],
        'hypothesis':'Removing unverified mirror evidence at missing-real/recovery boundary changes local angular acceleration.',
        'comparison':'Matched control/ablation with same reservation; compare local acceleration, global motion, contact and old heldout regression.',
        'validation_limits':'Reserved frames have new labels pending. Earlier runs used their automatic points. Not globally unseen data; do not call old heldout or pending new labels an independent acceptance test. No publication.'})
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(prepare(a.source,a.output))
