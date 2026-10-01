"""Preserve an accepted reference when new direction fitting regresses image agreement."""
from pathlib import Path
import json,shutil
from audit_racket_direction import audit as direction_audit
from audit_racket_motion import audit as motion_audit
from run_records import write,sha256

def compare(before,after,before_motion,after_motion):
    failures=[]
    for key,stat in [('image_shaft_error_deg','p95'),('head_center_error_canonical_px','p95'),('head_center_error_canonical_px','median')]:
        a=before[key][stat];b=after[key][stat]
        if a is not None and (b is None or b>a*1.05+1):failures.append(key+'_'+stat)
    if after_motion['acceleration_p95_deg']>before_motion['acceleration_p95_deg']*1.05+.1:failures.append('angular_acceleration')
    if after_motion['hidden']>before_motion['hidden']:failures.append('missing_frames')
    return failures

def gate(out):
    out=Path(out);poses=out/'racket_poses.json';candidate=out/'racket_poses_directional.json';shutil.copy2(poses,candidate)
    report={'status':'provisional_no_reference','candidate_sha256':sha256(candidate),'physical_face_identity_requires_review':True}
    reference=out/'racket_poses_reference.json'
    if reference.exists():
        before=direction_audit(reference,out/'racket_keypoints.json',out/'mesh_meta.json');after=direction_audit(candidate,out/'racket_keypoints.json',out/'mesh_meta.json');bm=motion_audit(reference);am=motion_audit(candidate);failures=compare(before,after,bm,am)
        report.update(status='needs_review' if failures else 'accepted',reference_sha256=sha256(reference),before=before,after=after,before_motion=bm,after_motion=am,failed_gates=failures)
        if failures:shutil.copy2(reference,poses)
    report['published_sha256']=sha256(poses);write(out/'racket_quality_gate.json',report)
    return report
