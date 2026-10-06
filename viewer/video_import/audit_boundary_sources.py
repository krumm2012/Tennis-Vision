"""Paired discovery-source rejection audit on a fixed unresolved UV population."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_shoulder_evidence import support_summary
from semantic_body_layers import sample_layers, distances
from run_records import sha256, write


def audit(original, refined):
    with np.load(original / 'additional_evidence.npz') as z:
        old={k:z[k] for k in z.files}
    rows=json.loads((original / 'frames.json').read_text())
    labels=[]; changes=[]; bindings={}
    for index,row in enumerate(rows):
        src=original/'semantic'/(row['key']+'.npz')
        path=refined/'semantic_discovery'/(row['key']+'.npz')
        rec=json.loads(path.with_suffix('.json').read_text())
        if rec['cache_sha256']!=sha256(path) or rec['source_cache_sha256']!=sha256(src):
            raise ValueError('Source binding changed')
        with np.load(src) as s:xy=s['xy'];box=s['box']
        with np.load(path) as d:
            group,conf,dist=sample_layers(d['labels'],d['confidence'],distances(d['labels']),xy,box)
        keep=(group==old['labels'][index]) & (conf>=.85) & (dist>=3)
        updated=np.where(keep,old['labels'][index],0).astype(np.uint8)
        if np.any((updated!=0)&(updated!=old['labels'][index])):raise ValueError('New material label invented')
        labels.append(updated);bindings[str(path.resolve())]=rec['cache_sha256']
        changes.append(dict(key=row['key'],before=int(np.isin(old['labels'][index],[2,3]).sum()),
                            after=int(np.isin(updated,[2,3]).sum())))
    labels=np.stack(labels);support=support_summary(labels,old['keys'])
    before=np.isin(old['labels'],[2,3]).any(0);after=np.isin(labels,[2,3]).any(0)
    summary=dict(population=len(before),frames=len(set(map(tuple,old['keys'][:,:2]))),views=len(labels),
        any_clear_before=int(before.sum()),any_clear_after=int(after.sum()),
        strong_before=int(old['strong'].sum()),strong_after=int(support['strong'].sum()),
        old_zero_clear_before=int((before&(old['old_count']==0)).sum()),
        old_zero_clear_after=int((after&(old['old_count']==0)).sum()),
        rejected_clear_observations=int((np.isin(old['labels'],[2,3])&~np.isin(labels,[2,3])).sum()),
        training_expanded=False,texture_repaired=False,
        interpretation='Rejected risky hypotheses, not improved independent reconstruction accuracy.')
    np.savez_compressed(refined/'guarded_discovery_evidence.npz',labels=labels,keys=old['keys'],
        ys=old['ys'],xs=old['xs'],face=old['face'],old_count=old['old_count'],**support)
    write(refined/'discovery_recheck.json',dict(summary=summary,per_view=changes,source_sha256=bindings))
    checks=[]
    for point in json.loads((original/'source_pixel_spot_checks.json').read_text()):
        with np.load(refined/'semantic_discovery'/(point['key']+'.npz')) as d:
            x,y=point['crop_xy'];label=int(d['labels'][y,x])
            checks.append(dict(point,refined_group=label,rejected_as_material_source=label==0))
    write(refined/'source_pixel_recheck.json',checks)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--refined',type=Path,required=True)
    a=p.parse_args();audit(a.original,a.refined)
