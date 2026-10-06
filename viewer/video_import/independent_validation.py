"""Prepare source-bound grip and two-coach validation packets without inventing labels.

The packet keeps model predictions away from the blank measurement/coach forms.
Agreement is computed only for confirmed, identically scoped paired ratings.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from practice_review import load, bindings
from practice_scoring import POLICY, POLICY_VERSION, POLICY_SHA256
from run_records import sha256, write

DIMENSIONS=[x['id'] for x in POLICY['dimensions']]


def coach_agreement(first, second):
    if first.get('policy_sha256') != POLICY_SHA256 or second.get('policy_sha256') != POLICY_SHA256:
        raise ValueError('Different scoring policy')
    a,b=first.get('reviewer','').strip(),second.get('reviewer','').strip()
    if not a or not b or a.casefold()==b.casefold():
        return dict(status='needs_two_independent_reviewers',paired_shots=0,dimension_mae=None)
    errors=[]; skipped=[]
    def indexed(doc):
        result={}
        for row in doc['shots']:
            if row['id'] in result:raise ValueError('Duplicate shot identity')
            result[row['id']]=row
        return result
    left,right=indexed(first),indexed(second)
    for ident in sorted(set(left)|set(right)):
        x,y=left.get(ident),right.get(ident)
        if x is None or y is None or x.get('confirmed') is not True or y.get('confirmed') is not True:
            skipped.append(dict(id=ident,reason='missing_or_unconfirmed'));continue
        same=all(x.get(k)==y.get(k) for k in ['video_sha256','evidence_sha256','video_frames','start_frame','end_frame','stroke','goal','machine'])
        bound=bool(x.get('video_sha256')) and isinstance(x.get('evidence_sha256'),dict) and bool(x['evidence_sha256'])
        interval=(type(x.get('video_frames')) is int and type(x.get('start_frame')) is int and type(x.get('end_frame')) is int and
                  0<=x['start_frame']<x['end_frame']<x['video_frames'])
        conditions=bool(x.get('goal')) and x.get('stroke') in ['Forehand','Backhand','Two-Handed Backhand','Volley','Serve'] and all(
            isinstance(x.get('machine',{}).get(k),str) and x['machine'][k].strip()
            for k in ['speed','frequency','direction','spin'])
        if not same or not bound or not interval or not conditions:
            skipped.append(dict(id=ident,reason='unmatched_or_incomplete_scope'));continue
        values=[[r.get('ratings',{}).get(k) for k in DIMENSIONS] for r in [x,y]]
        if not all(type(v) is int and 1<=v<=5 for row in values for v in row):
            skipped.append(dict(id=ident,reason='missing_or_invalid_ratings'));continue
        errors.append(np.abs(np.asarray(values[0])-values[1]))
    mae=dict(zip(DIMENSIONS,np.stack(errors).mean(0).tolist())) if errors else None
    return dict(status='paired_agreement_diagnostic' if errors else 'needs_confirmed_paired_shots',
        paired_shots=len(errors),dimension_mae=mae,skipped=skipped,
        accuracy_accepted=False,limits='Agreement is not validity or a calibrated acceptance threshold.')


def prepare(protocol, library, collection, reference, out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'prepare_code_snapshot.py')
    rows=json.loads(protocol.read_text())['entries']; statuses=[]; shots=[]; measurements=[]; prediction_index=[]
    for entry in rows:
        dataset=entry['viewer'].split('/')[2]; result=library/dataset/'result'
        state=load(result);name=entry['name']
        if state['video_sha256']!=entry['video_sha256']:raise ValueError('Video identity changed')
        prediction_path=(reference if name=='48.43' else collection/'clips'/name/'automatic_distance')/'predictions.json'
        prediction=json.loads(prediction_path.read_text())
        if prediction['video_sha256']!=state['video_sha256']:raise ValueError('Prediction video differs')
        if any(prediction.get(k) is not False for k in ['manual_labels_used','prescribed_grip_used','current_racket_poses_used']):
            raise ValueError('Prediction independence flags absent or invalid')
        declared=dict(video_sha256=state['video_sha256'],evidence_sha256=state['evidence_sha256'],
                      policy_sha256=POLICY_SHA256)
        shots.append(dict(id=name+'_shot_pending',clip=name,video_frames=state['frames'],**declared,
            viewer='http://127.0.0.1:18769'+entry['viewer'], start_frame=None,end_frame=None,
            stroke='Unknown',goal='',machine={k:'' for k in ['speed','frequency','direction','spin']},
            ratings={k:None for k in DIMENSIONS},confirmed=False,observation=''))
        usable=[r for r in prediction['frames'] if r['ratio']['usable']]
        chosen=[]
        if usable:
            # Spread across the usable timeline, rather than selecting only easiest errors.
            for i in np.unique(np.rint(np.linspace(0,len(usable)-1,min(3,len(usable)))).astype(int)):
                r=usable[i];chosen.append(r['frame'])
                measurements.append(dict(clip=name,video_sha256=state['video_sha256'],frame_0based=r['frame'],
                    original_source_frame=r['source_frame'],grip_definition='distance along handle axis from butt center to agreed grip anchor',
                    measured_distance_cm=None,measurement_method='',measurement_evidence='',
                    matched_same_grip_confirmed=False,reviewer='',approved=False))
        # Model outputs are deliberately kept in a separate file from blank reference forms.
        prediction_index.append(dict(clip=name,path=str(prediction_path.resolve()),sha256=sha256(prediction_path),
            eligible_frames=chosen,video_sha256=prediction['video_sha256']))
        landmarks=result/'racket_landmarks.json'
        labels=json.loads(landmarks.read_text()) if landmarks.exists() else {}
        actual_reviews=state['document']['shots']
        acceptance_path=result/'manual_grip_face_acceptance.json'
        acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
        accepted_files=acceptance.get('bound_files',{})
        accepted_binding_matches=bool(accepted_files) and all(
            (result/file).is_file() and sha256(result/file)==digest
            for file,digest in accepted_files.items())
        statuses.append(dict(clip=name,dataset=dataset,total_frames=state['frames'],
            current_evidence_sha256=bindings(result),manual_landmark_frames=len(labels.get('frames',[])),
            saved_coach_shots=len(actual_reviews),confirmed_coach_shots=sum(s.get('confirmed') is True for s in actual_reviews),
            ratio_usable_frames=len(usable),hand_axis_usable_frames=sum(r['hand_axis']['usable'] for r in prediction['frames']),
            physical_reference_available=False,
            hand_contact_face_acceptance='user_accepted_bound_scope' if accepted_binding_matches else 'pending_user_review',
            existing_acceptance_binding_matches=accepted_binding_matches))
    packet=dict(schema_version=1,policy_version=POLICY_VERSION,policy_sha256=POLICY_SHA256,
                reviewer='',instructions='Independently rate the same frozen shot; leave unknown values blank.',shots=shots)
    for reviewer in ['A','B']:write(out/f'coach_{reviewer}.json',dict(deepcopy(packet),reviewer_slot=reviewer))
    write(out/'grip_measurements.json',dict(schema_version=1,references=measurements,
        note='9 cm is prescribed, not measured. Do not copy model predictions as reference measurements.'))
    write(out/'prediction_index.json',prediction_index);write(out/'readiness.json',statuses)
    write(out/'coach_policy.json',POLICY)
    write(out/'coach_agreement.json',coach_agreement(packet,packet))
    write(out/'summary.json',dict(clips=len(rows),measurement_frames=len(measurements),
        confirmed_coach_shots=sum(r['confirmed_coach_shots'] for r in statuses),
        physical_measurements=0,automatic_distance_accuracy_accepted=False,
        other_eight_grip_clips_accepted=False,coach_scoring_calibrated=False,
        live_files_modified=False,code_sha256=sha256(Path(__file__)),protocol_sha256=sha256(protocol)))


def matched_prediction_source(dataset, prediction_path, prediction):
    """Resolve derived Viewer versus staged clip only with both video hashes."""
    for folder in [dataset,prediction_path.parent.parent]:
        normalized=folder/'source.mp4';raw=folder/'original.video'
        if not raw.exists():raw=normalized
        if (normalized.is_file() and raw.is_file() and
            sha256(normalized)==prediction['video_sha256'] and
            sha256(raw)==prediction['inputs_sha256']['video']):
            return raw
    raise ValueError('No hash-matched original and normalized prediction sources')


def export_crops(packet, protocol, library):
    entries={r['name']:r for r in json.loads(protocol.read_text())['entries']}
    dest=packet/'source_crops'
    if dest.exists():dest=packet/'source_crops_retry'
    dest.mkdir(exist_ok=False)
    records=[]
    for item in json.loads((packet/'prediction_index.json').read_text()):
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Predictions changed')
        prediction=json.loads(path.read_text());entry=entries[item['clip']]
        dataset=library/entry['viewer'].split('/')[2]
        # Derived Viewer datasets only have result/video.mp4. The frozen prediction
        # may instead belong to the adjacent staged clip used for that prediction.
        source=matched_prediction_source(dataset,path,prediction)
        cap=cv2.VideoCapture(str(source))
        try:
            for frame in item['eligible_frames']:
                row=prediction['frames'][frame]
                if row['frame']!=frame:raise ValueError('Prediction frame sequence differs')
                cap.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame']);ok,image=cap.read()
                if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=row['source_frame']+1:raise ValueError('Decode mismatch')
                scale=np.asarray(image.shape[1::-1])/prediction['image_size']
                points=np.asarray([row['grip_center_original_px']]+list(row['racket']['points'].values()))*scale
                lo=np.maximum(np.floor(points.min(0)).astype(int)-35,0)
                hi=np.minimum(np.ceil(points.max(0)).astype(int)+36,image.shape[1::-1])
                crop=image[lo[1]:hi[1],lo[0]:hi[0]]
                path=dest/f"{item['clip']}_f{frame:03}.png";cv2.imwrite(str(path),crop)
                records.append(dict(clip=item['clip'],frame_0based=frame,original_frame_0based=row['source_frame'],
                    original_size=list(image.shape[1::-1]),normalized_size=prediction['image_size'],
                    source_sha256=prediction['inputs_sha256']['video'],normalized_sha256=prediction['video_sha256'],
                    native_crop_xyxy=np.r_[lo,hi].tolist(),image=str(path.relative_to(packet)),image_sha256=sha256(path)))
        finally:cap.release()
    write(packet/'crop_manifest.json',dict(role='unlabelled native crops; automatic selection, not blind or measured truth',
        rows=records,code_sha256=sha256(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for k in ['protocol','library','collection','reference','output']:prep.add_argument('--'+k,type=Path,required=True)
    audit=sub.add_parser('compare-coaches')
    for k in ['first','second','output']:audit.add_argument('--'+k,type=Path,required=True)
    crops=sub.add_parser('export-crops')
    for k in ['packet','protocol','library']:crops.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.protocol,a.library,a.collection,a.reference,a.output)
    elif a.action=='export-crops':export_crops(a.packet,a.protocol,a.library)
    else:write(a.output,coach_agreement(json.loads(a.first.read_text()),json.loads(a.second.read_text())))
