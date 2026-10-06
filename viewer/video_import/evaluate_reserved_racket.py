"""Evaluate newly reviewed reserved labels without fitting or replacing old gates."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from racket_landmarks import validate
from audit_racket_evidence_boundaries import NAMES,projections,residuals
from audit_racket_roles import ray_pair
from run_records import sha256,write


def reservation_gate(labels,protocol,report):
    frames={r['frame'] for r in labels['frames']}
    if not frames or not frames<=set(protocol['reserved_frames']):raise ValueError('Labels outside frozen reservation')
    if frames&set(report['train_frames']):raise ValueError('Reserved labels contaminated by training')
    if labels['video_sha256']!=protocol['video_sha256'] or labels['video_sha256']!=report['video_sha256']:raise ValueError('Video identity mismatch')


def stats(values):
    x=np.array(values,float)
    return {'samples':len(x),'median':float(np.median(x)) if len(x) else None,
            'mean':float(x.mean()) if len(x) else None,'p95':float(np.percentile(x,95)) if len(x) else None}


def evaluate(labels_path,experiment,video,output):
    if output.exists():raise ValueError('Use fresh evaluation output')
    protocol=json.loads((experiment/'protocol.json').read_text());reference=experiment/'control/staged/result'
    meta=json.loads((reference/'mesh_meta.json').read_text());labels=validate(json.loads(labels_path.read_text()),meta)
    if sha256(video)!=meta['video_sha256']:raise ValueError('Source video mismatch')
    mirror=json.loads((reference/'mirror_geometry.json').read_text());model=json.loads((reference/'racket_poses_directional.json').read_text())['model']
    reports={name:json.loads((experiment/name/'gpu-fit/refit_report.json').read_text()) for name in ['control','mirror_ablation']}
    predictions={};runs={};scale=meta['image_size'][0]/1280
    for name,report in reports.items():
        reservation_gate(labels,protocol,report)
        staged=experiment/name/'staged/result'
        for file in ['mesh_meta.json','mirror_geometry.json','racket_poses_directional.json']:
            if sha256(staged/file)!=sha256(reference/file):raise ValueError('Candidate geometry/camera metadata differs')
        candidate=experiment/name/'gpu-fit/mhr_refit_candidate.npz'
        if sha256(candidate)!=report['candidate_sha256']:raise ValueError('Candidate hash differs')
        with np.load(candidate,allow_pickle=False) as d:predictions[name]=(d['racket_rotation'].copy(),d['racket_translation'].copy())
    prior=json.loads((reference/'racket_poses_directional.json').read_text())['frames']
    predictions['initial']=(np.array([r['rotation_camera_columns'] for r in prior]),np.array([r['translation_camera_m'] for r in prior]))
    projected={}
    for name,(rotation,translation) in predictions.items():
        rows=[];errors={view:[] for view in ['points','mirror_points']};angles={view:[] for view in errors};by_part={view:{} for view in errors}
        projected[name]={}
        for row in labels['frames']:
            i=row['frame'];record={'frame':i,'frame_1based':i+1,'views':{}};projected[name][i]={}
            for view in errors:
                uv=projections(rotation[i],translation[i],model,meta,i,mirror if view=='mirror_points' else None);projected[name][i][view]=uv
                result=residuals(uv,row,view,scale);record['views'][view]=result
                for part,error in result['point_error_canonical_px'].items():
                    errors[view].append(error);by_part[view].setdefault(part,[]).append(error)
                if result['shaft_error_deg'] is not None:angles[view].append(result['shaft_error_deg'])
            rows.append(record)
        runs[name]={'records':rows,'summary':{view:{'point_error_canonical_px':stats(errors[view]),'shaft_error_deg':stats(angles[view]),'by_part':{part:stats(v) for part,v in by_part[view].items()}} for view in errors}}
    output.mkdir(parents=True);shutil.copy2(labels_path,output/'labels_received.json');write(output/'labels_validated.json',labels)
    geometry=[]
    for row in labels['frames']:
        for part in set(row['points'])&set(row['mirror_points']):
            if part.startswith('rim_'):continue
            result=ray_pair(row['points'][part],row['mirror_points'][part],meta['focal'][row['frame']],meta['image_size'],mirror['normal_camera'],mirror['distance_camera_m'])
            geometry.append({'frame':row['frame'],'point':part,**result})
    cap=cv2.VideoCapture(str(video));sheets=[]
    try:
        for row in labels['frames']:
            i=row['frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,source=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))!=i+1:raise ValueError('Decode timeline mismatch')
            tiles=[]
            for view in ['points','mirror_points']:
                seen=row[view]
                if not seen:continue
                positions=np.array(list(seen.values()));lo=np.maximum(positions.min(0).astype(int)-70,0);hi=np.minimum(positions.max(0).astype(int)+70,source.shape[1::-1])
                for name in ['control','mirror_ablation']:
                    canvas=source.copy();uv=projected[name][i][view]
                    cv2.polylines(canvas,[np.rint(uv[[0,1,2]]).astype(np.int32)],False,(255,150,30),3)
                    for part,p in seen.items():
                        cv2.circle(canvas,tuple(np.rint(p).astype(int)),6,(30,240,30),-1)
                    crop=canvas[lo[1]:hi[1],lo[0]:hi[0]];ratio=min(500/crop.shape[1],300/crop.shape[0]);resized=cv2.resize(crop,None,fx=ratio,fy=ratio);tile=np.zeros((340,500,3),np.uint8);tile[:resized.shape[0],:resized.shape[1]]=resized
                    cv2.putText(tile,f'UI{i+1} {view} {name}',(8,323),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);tiles.append(tile)
            file=output/f'frame_{i:04d}_comparison.jpg';cv2.imwrite(str(file),np.concatenate(tiles,axis=1));sheets.append({'frame':i,'path':str(file.resolve()),'sha256':sha256(file)})
    finally:cap.release()
    result={'status':'new_reserved_labels_evaluated_not_accepted','labels_sha256':sha256(labels_path),'video_sha256':sha256(video),'protocol_sha256':sha256(experiment/'protocol.json'),
            'frames':[r['frame'] for r in labels['frames']],'runs':runs,'mirror_ray_diagnostics':geometry,'sheets':sheets,'code_sha256':sha256(Path(__file__)),
            'candidate_report_sha256':{name:sha256(experiment/name/'gpu-fit/refit_report.json') for name in reports},'candidate_sha256':{name:report['candidate_sha256'] for name,report in reports.items()},
            'published':False,'labels_used_for_training':False,'limits':'Labels received after frozen A/B fitting and excluded in those runs. Earlier historical runs used automatic points at these frames, so not globally unseen data. L2 point residuals normalized to 1280px width; unordered rim pair chooses minimum-sum diagnostic assignment. Camera/plane and racket dimensions are estimates; no physical face/bevel or skin contact validation.'}
    write(output/'reserved_label_evaluation.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['labels','experiment','video','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=evaluate(a.labels,a.experiment,a.video,a.output);print(json.dumps({name:run['summary'] for name,run in r['runs'].items()}))
