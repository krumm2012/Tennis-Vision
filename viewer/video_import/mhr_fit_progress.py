"""Evidence-bound completion records; a completed optimiser is not an accepted fit.

Per-output completion.json is a current snapshot. Content-addressed ledger events
are immutable and idempotent, so failure, pending transport and later recovery
remain visible. Native replay audits are a different event kind from joint fits.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path

from run_records import now, sha256, write

DEFAULT_LEDGER = Path(__file__).resolve().parents[2]/'output/mhr_fit_progress'
FILES = ['progress_attempt.json', 'remote_job.json', 'request.json', 'readiness.json', 'refit_report.json',
         'remote_manifest.json', 'run_manifest.json', 'contact_comparison.json', 'worker.log']


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _json_safe(value):
    # Preserve an invalid report's actual SHA while making the ledger explicit
    # about invalid numeric values instead of emitting non-standard JSON NaN.
    if isinstance(value,float) and not math.isfinite(value): return {'nonfinite_value':str(value)}
    if isinstance(value,dict): return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,list): return [_json_safe(v) for v in value]
    return value


def _stage(state='unknown', evidence=None, **values):
    return {'state': state, 'evidence': evidence or [], **values}


def _flag(value):
    return 'passed' if value is True else 'failed' if value is False else 'unknown'


def _recover_context(out):
    """Use the current attempt anchor, never an old job left in a reused folder."""
    def read(name):
        try: return json.loads((out/name).read_text())
        except (ValueError,OSError): return {}
    anchor, snapshot = read('progress_attempt.json'), read('completion.json')
    run_id=anchor.get('run_id','')
    if anchor.get('output')!=str(out) or len(run_id)!=32 or any(ch not in '0123456789abcdef' for ch in run_id):
        return {'current_files':['progress_attempt.json'],'phase':'invalid_attempt_anchor','verification_failed':True} if (out/'progress_attempt.json').exists() else {}
    context={'run_id':run_id,'current_files':['progress_attempt.json'],'phase':'manual_recovery',
             'remote_job_id':anchor.get('remote_job_id')}
    snapshot_bound=(snapshot.get('output')==str(out) and snapshot.get('run_id')==run_id and snapshot.get('event_kind')=='joint_fit')
    remote_job=read('remote_job.json')
    job_bound=bool(anchor.get('remote_job_id') and remote_job.get('job_id')==anchor['remote_job_id'])
    if snapshot_bound:
        execution=snapshot.get('execution',{})
        context.update(phase=execution.get('phase','manual_recovery'),saved_execution_state=execution.get('state'),
                       parent_job_id=snapshot.get('parent_job_id'),remote_status=execution.get('remote_state'),
                       dispatch_attempted=execution.get('dispatch_attempted'),dispatch_acknowledged=execution.get('dispatch_acknowledged'),
                       error=execution.get('error'),progress_recording_errors=execution.get('progress_recording_errors',[]))
        context['current_files']=[k for k in snapshot.get('provenance',{}).get('source_files',{}) if k in FILES]
        # A no-dispatch attempt's snapshot deliberately excludes older job files.
        if not anchor.get('remote_job_id'):
            context['current_files']=[k for k in context['current_files'] if k in ['progress_attempt.json','readiness.json']]
    if job_bound:
        context['current_files']+=['remote_job.json','request.json','readiness.json']
        manifest=read('remote_manifest.json') or read('run_manifest.json')
        if manifest.get('remote_job_id')==anchor['remote_job_id']:
            context['current_files']+=['remote_manifest.json','run_manifest.json','refit_report.json',
                                       'mhr_refit_candidate.npz','contact_comparison.json','worker.log']
            context['remote_status']=manifest.get('status')
        elif snapshot_bound:
            if snapshot.get('provenance',{}).get('candidate_sha256'):
                context['current_files'].append('mhr_refit_candidate.npz')
    elif anchor.get('remote_job_id'):
        context.update(current_files=['progress_attempt.json'],verification_failed=True,
                       error={'type':'IdentityMismatch','message':'Current attempt anchor does not match remote_job.json.'})
    return context


def collect(output, context=None, execution_state=None, phase=None):
    """Build a fit snapshot from actual files; unknown evidence never becomes passed."""
    out = Path(output).resolve(); context = _recover_context(out) if context is None else context
    documents, evidence, read_errors = {}, {}, {}
    for name in FILES:
        if 'current_files' in context and name not in context['current_files']: continue
        path = out/name
        if not path.exists(): continue
        evidence[name] = {'path': str(path), 'sha256': sha256(path)}
        if name.endswith('.json'):
            try: documents[name] = json.loads(path.read_text())
            except (ValueError, OSError) as error: read_errors[name] = type(error).__name__
    local_readiness = documents.get('readiness.json', {})
    manifest = documents.get('remote_manifest.json', documents.get('run_manifest.json', {}))
    readiness = manifest.get('readiness') or local_readiness
    report = documents.get('refit_report.json', manifest.get('refit', {}))
    job = documents.get('remote_job.json', {})
    request = documents.get('request.json', {})
    refit_evidence = ['refit_report.json'] if 'refit_report.json' in documents else ['remote_manifest.json'] if report else []
    run_id = context.get('run_id') or job.get('job_id') or manifest.get('remote_job_id')
    if not run_id:
        # Stable identity for historical needs-input folders that never dispatched.
        run_id = 'local-'+hashlib.sha256(str(out).encode()).hexdigest()[:32]
    candidate = out/'mhr_refit_candidate.npz'
    candidate_sha = sha256(candidate) if candidate.exists() and ('current_files' not in context or 'mhr_refit_candidate.npz' in context['current_files']) else None
    candidate_expected = report.get('candidate_sha256') or manifest.get('artifact_sha256', {}).get('mhr_refit_candidate.npz')
    hash_values = {'video': [v for v in [readiness.get('video_sha256'), report.get('video_sha256'),
                                       request.get('source_sha256'), manifest.get('source_sha256'), context.get('video_sha256')] if v],
                   'archive': [v for v in [readiness.get('archive_sha256'), report.get('source_archive_sha256'),
                                         request.get('archive_sha256'), context.get('archive_sha256')] if v]}
    hash_conflicts = {k:sorted(set(v)) for k,v in hash_values.items() if len(set(v)) > 1}
    log = (out/'worker.log').read_text(errors='replace') if 'worker.log' in evidence else ''
    camera_failed = '重推理相机与现有标定不一致' in log
    remote_state = context.get('remote_status') or manifest.get('status')
    computed = report.get('full_body_joint_fit_completed') is True
    state = execution_state or ('completed' if computed else 'failed' if remote_state=='failed' else
                                'needs_input' if remote_state=='needs_input' or readiness.get('ready') is False else
                                'pending' if remote_state in ['queued','generating'] else context.get('saved_execution_state') or 'unknown')
    stages = {
        'input_readiness': _stage(_flag(readiness.get('ready')), ['readiness.json'], blocked_by=readiness.get('blocked_by', [])),
        'joint_optimisation': _stage('completed' if computed else 'not_completed' if state in ['failed','needs_input'] else 'unknown', refit_evidence, steps=report.get('steps')),
        'camera_consistency': _stage('failed' if camera_failed else _flag(report.get('camera_roots_preserved')), ['worker.log'] if camera_failed else refit_evidence),
        'shared_identity_optimization': _stage('passed' if report.get('shared_identity_optimization_completed') is True else 'not_completed' if report.get('shape_and_scale_preserved') is True else 'unknown', refit_evidence, shape_and_scales_locked=report.get('shape_and_scale_preserved')),
        'camera_and_scale_calibration': _stage('passed' if report.get('camera_and_scale_calibration_completed') is True else 'not_completed' if readiness.get('camera_calibrated') is False or report.get('camera_roots_preserved') is True else 'unknown', refit_evidence+['readiness.json'], camera_roots_fixed=report.get('camera_roots_preserved')),
        'native_replay': _stage(evidence=refit_evidence),
        'body_displacement_regression': _stage(evidence=refit_evidence, p95_m=report.get('body_displacement_p95_m'), threshold_m=.03),
        'heldout_landmark_regression': _stage(evidence=refit_evidence, before_canonical_px=report.get('heldout_landmark_before_canonical_px'), after_canonical_px=report.get('heldout_landmark_after_canonical_px'), required_ratio_below=.9),
        'numerical_regression': _stage(_flag(report.get('numerical_regression_gate_passed')), refit_evidence),
        'racket_fit_validation': _stage(_flag(report.get('racket_gate_passed')), refit_evidence),
        'hand_mesh_contact': _stage('passed' if report.get('mesh_contact_completed') is True and report.get('mesh_contact_gate_passed') is True else 'failed' if report.get('mesh_contact_gate_passed') is False else 'not_completed' if report and report.get('mesh_contact_completed') is not True else 'unknown', refit_evidence, method=report.get('contact_method')),
        'independent_observations': _stage('passed' if report.get('independent_observations_verified') is True else 'not_completed' if (report.get('validation_source') or readiness.get('validation_source'))=='automatic_contours_unverified' else 'unknown', refit_evidence+['readiness.json']),
        'measured_dimensions': _stage('passed' if report.get('dimensions_measured', readiness.get('size_measured')) is True else 'not_completed' if report.get('dimensions_measured', readiness.get('size_measured')) is False else 'unknown', refit_evidence+['readiness.json']),
        'physical_grip_bevel': _stage('passed' if report.get('physical_grip_bevel_verified') is True else 'not_completed' if report.get('physical_grip_bevel_verified') is False else 'unknown', refit_evidence),
        'visual_review': _stage(_flag(report.get('visual_review_passed')), refit_evidence),
        'candidate_integrity': _stage('passed' if candidate_sha and candidate_sha==candidate_expected and not hash_conflicts else 'failed' if (candidate_sha and candidate_expected and candidate_sha!=candidate_expected) or hash_conflicts or context.get('verification_failed') else 'unknown', refit_evidence, actual_sha256=candidate_sha, expected_sha256=candidate_expected, source_hash_conflicts=hash_conflicts),
    }
    if report.get('native_replay_mesh_max_m') is not None and report.get('native_replay_joint_max_m') is not None:
        values = [report['native_replay_mesh_max_m'], report['native_replay_joint_max_m']]
        stages['native_replay'].update(state='passed' if all(0<=v<=1e-4 for v in values) else 'failed', mesh_max_m=values[0], joint_max_m=values[1], threshold_m=1e-4)
    p95 = report.get('body_displacement_p95_m')
    if isinstance(p95, (float,int)):
        stages['body_displacement_regression']['state'] = 'passed' if 0<=p95<.03 else 'failed'
    before, after = report.get('heldout_landmark_before_canonical_px'), report.get('heldout_landmark_after_canonical_px')
    if isinstance(before,(float,int)) and isinstance(after,(float,int)):
        stages['heldout_landmark_regression']['state'] = 'passed' if before>0 and 0<=after<before*.9 else 'failed'
    acceptance = 'rejected' if state=='failed' or any(s['state']=='failed' for s in stages.values()) else 'accepted_joint_fit' if computed and all(s['state'] in ['passed','completed'] for s in stages.values()) else 'needs_review' if computed else 'not_assessed'
    unfinished = [name for name,value in stages.items() if value['state'] not in ['passed','completed']]
    suggestions = {'camera_consistency':'Repack source roots/focal and observation coordinates from the same native generation before fitting.',
                   'input_readiness':'Complete real/mirror observations and an independent train/heldout split.',
                   'native_replay':'Verify the exact MHR model, native parameter layout and camera-axis conversion.',
                   'shared_identity_optimization':'Estimate and validate one shared identity shape/scale from multiple videos; locked per-frame shape is not joint identity optimisation.',
                   'camera_and_scale_calibration':'Calibrate camera, physical scale and mirror geometry with independent anchors; preserving predicted camera roots is not calibration.',
                   'heldout_landmark_regression':'Correct handle keypoints/directed racket face observations; rerun against the same fixed heldout observations.',
                   'numerical_regression':'Improve the candidate without changing the acceptance threshold or validation observations.',
                   'racket_fit_validation':'Validate handle/frame direction and temporal jitter against independent racket observations.',
                   'hand_mesh_contact':'Implement hand mesh-to-handle surface contact and penetration checks; joint/cylinder proxies do not complete this stage.',
                   'independent_observations':'Review clear real/mirror keyframes; automatic silhouettes are not independently verified labels.',
                   'measured_dimensions':'Measure racket dimensions or retain an explicit assumed-size limitation.',
                   'physical_grip_bevel':'Verify grip bevel/contact from clear views; do not infer certainty from silhouettes.',
                   'visual_review':'Review grip and racket from front/back/left/right/oblique views.',
                   'candidate_integrity':'Retrieve and verify the candidate, report and exact source hashes.',
                   'joint_optimisation':'Run the bounded GPU joint fit after the prerequisites pass.',
                   'body_displacement_regression':'Check anatomy/body displacement against the source and independent body observations.'}
    return {'schema_version':1, 'event_kind':'joint_fit', 'run_id':run_id, 'output':str(out),
            'remote_job_id':context.get('remote_job_id') or job.get('job_id') or manifest.get('remote_job_id'),
            'parent_job_id':context.get('parent_job_id') or job.get('parent_job') or request.get('base_job_id'),
            'execution':{'state':state, 'phase':phase or context.get('phase') or 'historical_backfill', 'remote_state':remote_state, 'computed_fit':computed,
                         'dispatch_attempted':context.get('dispatch_attempted'), 'dispatch_acknowledged':context.get('dispatch_acknowledged'), 'error':context.get('error'), 'remote_error_type':manifest.get('error_type'), 'progress_recording_errors':context.get('progress_recording_errors',[])},
            'acceptance':{'state':acceptance, 'accepted_joint_fit':acceptance=='accepted_joint_fit', 'published_to_viewer':report.get('published_to_viewer',False)},
            'provenance':{'video_sha256':hash_values['video'][0] if hash_values['video'] else None, 'source_archive_sha256':hash_values['archive'][0] if hash_values['archive'] else None,
                          'candidate_sha256':candidate_sha, 'candidate_expected_sha256':candidate_expected, 'fit_code_sha256':report.get('code_sha256') or manifest.get('code_sha256',{}).get('refit_fullbody.py'),
                          'local_driver_sha256':sha256(Path(__file__).with_name('run_fullbody_refit.py')),
                          'local_fit_script_sha256':sha256(Path(__file__).with_name('refit_fullbody.py')),
                          'git_commit':manifest.get('git_commit'), 'source_files':evidence, 'read_errors':read_errors},
            'readiness':readiness, 'local_readiness':local_readiness, 'readiness_source':'remote_manifest' if manifest.get('readiness') else 'readiness.json', 'dimension_status':'measured' if stages['measured_dimensions']['state']=='passed' else 'assumed' if readiness.get('assumed_dimensions_permitted') and stages['measured_dimensions']['state']=='not_completed' else 'unknown',
            'optimization_scope':{'skeletal_pose_and_racket_computed':computed, 'identity_shape_and_scales_locked':report.get('shape_and_scale_preserved'), 'camera_roots_locked':report.get('camera_roots_preserved')},
            'stages':stages, 'contact_comparison':documents.get('contact_comparison.json'),
            'unfinished_items':unfinished, 'next_steps':[suggestions[k] for k in unfinished],
            'metric_limits':'Execution completion and acceptance are separate. No aggregate fit percentage; regression gates and source replay are not independent body/grip ground truth.'}


def record(content, output=None, ledger=DEFAULT_LEDGER):
    """Append one immutable provenance event; repeat content yields the same event."""
    content = _json_safe(content)
    content['provenance']['progress_recorder_sha256'] = sha256(__file__)
    ledger = Path(ledger); ledger.mkdir(parents=True, exist_ok=True)
    event_id = hashlib.sha256(_canonical(content).encode()).hexdigest()
    events = ledger/'events'; events.mkdir(exist_ok=True)
    with (ledger/'ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = events/(event_id+'.json')
        if target.exists():
            event = json.loads(target.read_text())
            if _canonical(event['content']) != _canonical(content): raise ValueError('ledger event identity conflict')
        else:
            event = {'schema_version':1, 'event_id':event_id, 'recorded_at':now(), 'content':content}
            with target.open('x') as stream: stream.write(json.dumps(event,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        all_events = sorted((json.loads(p.read_text()) for p in events.glob('*.json')),key=lambda e:(e['recorded_at'],e['event_id']))
        runs = {}
        for e in all_events:
            c = e['content']; run = runs.setdefault(c['run_id'], {'run_id':c['run_id'], 'event_ids':[], 'event_kind':c['event_kind']})
            run['event_ids'].append(e['event_id']); run.update(latest_event_id=e['event_id'], latest_execution=c['execution']['state'], latest_acceptance=c['acceptance']['state'])
        write(ledger/'ledger.json', {'schema_version':1, 'events':[{k:e[k] for k in ['event_id','recorded_at']} for e in all_events], 'runs':list(runs.values()),
                                   'accepted_joint_fit_runs':sum(r['event_kind']=='joint_fit' and r['latest_acceptance']=='accepted_joint_fit' for r in runs.values())})
        snapshot = dict(content, event_id=event_id, recorded_at=event['recorded_at'], ledger_event=str(target.resolve()))
        if output is not None: write(Path(output)/'completion.json',snapshot)
    return snapshot


def record_fit(output, ledger=DEFAULT_LEDGER, **kwargs):
    return record(collect(output, **kwargs), output, ledger)


def record_native_audit(report_path, ledger=DEFAULT_LEDGER):
    path = Path(report_path).resolve(); report = json.loads(path.read_text())
    clips=report.get('clips',[])
    checks=[]
    for clip in clips:
        checks += [clip.get(k) for k in ['source_hash_matches_request','topology_matches_provisioned_model','embedded_video_hash_matches','frame_count_matches_manifest']]
        checks += [clip.get(view,{}).get('replay_gate_passed') for view in ['real','mirror']]
    failed=any(v is False for v in checks) or any(c.get('status') in ['source_identity_mismatch','audit_requires_review','invalid_source','replay_mismatch'] for c in clips)
    replay_state='failed' if failed else 'passed' if clips and all(v is True for v in checks) and all(c.get('status')=='verified_replay' for c in clips) else 'unknown'
    content = {'schema_version':1, 'event_kind':'native_replay_audit', 'run_id':report['audit_id'], 'output':str(path.parent),
               'execution':{'state':'completed' if report.get('status') in ['complete','requires_review'] else 'unknown'},
               'acceptance':{'state':'not_applicable', 'accepted_joint_fit':False, 'published_to_viewer':False},
               'provenance':{'report_path':str(path), 'report_sha256':sha256(path), 'script_sha256':report.get('script_sha256'), 'model_sha256':report.get('mhr_model_sha256'),
                             'sources':[{'job_id':c.get('job_id'),'video':c.get('video'),'archive_sha256':c.get('archive_sha256'),'video_sha256':c.get('normalized_video_sha256')} for c in clips]},
               'stages':{'native_replay_audit':_stage(replay_state, [str(path)],clips=len(clips)), 'joint_optimisation':_stage('not_completed')},
               'unfinished_items':['joint_optimisation_not_part_of_this_audit'], 'next_steps':[],
               'metric_limits':'Native replay audit verifies source consistency; it does not count as a joint fit or independent geometric validation.'}
    return record(content, path.parent, ledger)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path); p.add_argument('--audit-report',type=Path); p.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    a = p.parse_args()
    if bool(a.output)==bool(a.audit_report): p.error('choose one of --output or --audit-report')
    result = record_fit(a.output,a.ledger) if a.output else record_native_audit(a.audit_report,a.ledger)
    print(json.dumps({'run_id':result['run_id'],'event_id':result['event_id'],'execution':result['execution']['state'],'acceptance':result['acceptance']['state']},ensure_ascii=False))


if __name__=='__main__': main()
