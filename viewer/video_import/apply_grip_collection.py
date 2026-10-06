"""Sequential, video-scoped fixed-grip previews; preserve accepted motion.

Each clip keeps its native body/camera and its own automatic observations.
The accepted reference supplies definitions and the prescribed 9cm distance,
never rotations, translations, labels or acceptance for a different video.
"""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_records import now, sha256, write, revision

ROOT = Path(__file__).resolve().parents[2]
MODULES = Path(__file__).parent


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    elif sha256(source) != sha256(target):
        raise ValueError('Existing staged input differs: ' + str(target))


def identity(meta, video):
    if meta['video_sha256'] != sha256(video):
        raise ValueError('Normalized video identity differs')
    if meta['fps'] != 25 or meta['frames'] < 3:
        raise ValueError('Unsupported playback timeline')


def prepare(source, target, reference):
    meta = json.loads((source/'result/mesh_meta.json').read_text())
    identity(meta, source/'source.mp4')
    target.mkdir(parents=True, exist_ok=True)
    for name in ['source.mp4', 'original.video']:
        if (source/name).exists():
            link(source/name, target/name)
    record = json.loads((source/'record.json').read_text())
    write(target/'record.json', record)
    work = Path('attempts')/f"{record['attempt']:04d}"/'work'
    for name in ['reconstruction.npz', 'multiview_manifest.json']:
        if (source/work/name).exists():
            link(source/work/name, target/work/name)
    result = target/'result'
    result.mkdir(exist_ok=True)
    # Mutable JSON is copied; large, immutable binary/video/image inputs are linked.
    for p in (source/'result').iterdir():
        if not p.is_file() or p.name.startswith(('racket_', 'wilson_', 'grip_')):
            continue
        if p.suffix in ['.bin', '.png', '.mp4']:
            link(p, result/p.name)
        elif not (result/p.name).exists():
            shutil.copy2(p, result/p.name)
    write(target/'original_meta.json', meta)
    write(result/'racket_grip_settings.json', {
        **{k:meta[k] for k in ['video_sha256','image_size','fps']},
        'schema_version':1,'grip_from_butt_m':.09,'source':'user_specified',
        'distance_measured':False,'reference_acceptance_sha256':sha256(reference),
        'accepted_for_this_clip':False,
        'definitions':{'grip':'grasp-region center on handle axis',
                       'handle_end':'butt-cap axial center','throat':'lower string-bed edge'},
    })
    model = json.loads((reference.parent/'racket_poses.json').read_text())['model']
    ring = np.asarray(model['head_outline'])
    write(result/'racket_dimensions.json', {
        **{k:meta[k] for k in ['video_sha256','image_size','fps']},
        'schema_version':1,'measured':False,'grip_style':'unknown',
        'dimensions_cm':{'length':model['length_m']*100,
                         'head_width':2*model['head_half_width_m']*100,
                         'head_height':float(np.ptp(ring[:,1]))*100},
    })
    return meta


def filter_roles(stage):
    """Exclude the opposite tracked person's racket from real-only fitting."""
    from racket_keypoints import role_compatible
    sys.path.insert(0,str(ROOT/'viewer/sam3d'))
    from fit_racket_video import head_ellipse
    from fit_racket_pose import project
    result=stage/'result';meta=json.loads((stage/'original_meta.json').read_text())
    record=json.loads((stage/'record.json').read_text())
    archive=stage/'attempts'/f"{record['attempt']:04d}"/'work/reconstruction.npz'
    with np.load(archive,allow_pickle=False) as d:
        real=d['joints'][:,41]+d['source_roots']
        virtual=d['mirror_joints'][:,41]+d['mirror_roots']
        valid=d['mirror_valid'].copy();mf=d['mirror_focal'].copy()
    scale=meta['image_size'][0]/1280
    real_uv=[project(p[None],meta['focal'][i],meta['image_size'])[0] for i,p in enumerate(real)]
    mirror_uv=[project(p[None],mf[i],meta['image_size'])[0] if valid[i] and p[2]>.1 else None for i,p in enumerate(virtual)]
    rejected=0
    def compatible(poly,i):
        e=head_ellipse(np.asarray(poly)/scale)
        return e is not None and role_compatible(e[0]*scale,real_uv[i],mirror_uv[i],scale)
    for name in ['racket_candidates.json','racket_roi_candidates.json']:
        path=result/name;original=json.loads(path.read_text())
        write(stage/(name.replace('.json','_unfiltered.json')),original)
        payload=copy.deepcopy(original)
        for i,row in enumerate(payload['frames']):
            kept=[]
            for c in row['candidates']:
                if c.get('view')=='mirror' or not compatible(c['polygon'],i):
                    rejected+=1
                else:
                    c['view']='real';kept.append(c)
            row['candidates']=kept
        write(path,payload)
    path=result/'racket_poses.json';poses=json.loads(path.read_text())
    write(stage/'racket_seed_unfiltered.json',poses)
    for i,row in enumerate(poses['frames']):
        if row.get('observed_polygon') and not compatible(row['observed_polygon'],i):
            row.pop('observed_polygon');row.update(status='missing_observation',quality='hidden')
    write(path,poses)
    # Mirror estimates remain available for display and diagnostics, not racket loss.
    fit_meta=copy.deepcopy(meta);fit_meta['mirror_available']=False
    fit_meta['multiview_refined_available']=False
    write(result/'mesh_meta.json',fit_meta)
    return {'rejected_opposite_or_unusable_candidates':rejected,
            'mirror_racket_loss_enabled':False,'role_seed':'independent native real/mirror wrists',
            'reason':'no clip-specific held-out mirror racket validation'}


def command(stage, script, args):
    with (stage/(script+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-u',str(MODULES/script),*map(str,args)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def auto_distance(stage, model, device):
    """Freeze unconstrained predictions separately; no fitting poses/9cm inputs."""
    from audit_automatic_grip import predict,evaluate
    output=stage/'automatic_distance';output.mkdir()
    prediction=predict(stage,model,output,device)
    labels={**{k:prediction[k] for k in ['video_sha256','image_size','fps']},'frames':[]}
    report=evaluate(prediction,labels)
    report.update(status='awaiting_independent_reference',labels_available=False,
                  predictions_sha256=sha256(output/'predictions.json'),
                  video_sha256=prediction['video_sha256'],
                  prescribed_grip_used=False,current_racket_poses_used=False,
                  limits='No annotations or measured distance for this clip. Counts/estimates are diagnostics, not an accuracy score.')
    write(output/'report.json',report)
    return report


def audit(stage, roles):
    from audit_racket_motion import audit as motion
    from audit_racket_direction import audit as direction
    from grip_contact_audit import audit as contact
    result=stage/'result';meta=json.loads((stage/'original_meta.json').read_text())
    poses=json.loads((result/'racket_poses.json').read_text())
    for k in ['video_sha256','image_size','fps']:
        if poses[k]!=meta[k]:raise ValueError('Candidate identity differs')
    if len(poses['frames'])!=meta['frames'] or abs(poses['model']['grip_y_m']-.09)>1e-9:
        raise ValueError('Fixed 9cm setting not applied')
    rotations=np.asarray([r['rotation_camera_columns'] for r in poses['frames']])
    if not np.isfinite(rotations).all() or not np.allclose(np.linalg.det(rotations),1,atol=1e-6):
        raise ValueError('Invalid rigid racket rotation')
    record=json.loads((stage/'record.json').read_text())
    archive=stage/'attempts'/f"{record['attempt']:04d}"/'work/reconstruction.npz'
    with np.load(archive,allow_pickle=False) as d:
        contacts=contact(poses,d['joints'],d['source_roots'])
    write(result/'grip_contact_report.json',contacts)
    m=motion(result/'racket_poses.json')
    d=direction(result/'racket_poses.json',result/'racket_keypoints.json',result/'mesh_meta.json')
    def over(stat,limit):return stat is None or stat>limit
    failures=[]
    if m['step_max_deg']>45 or m['acceleration_p95_deg']>20:failures.append('angular_motion')
    if over(d['image_shaft_error_deg']['p95'],25):failures.append('image_shaft_alignment')
    if over(d['head_center_error_canonical_px']['p95'],40):failures.append('head_center_alignment')
    report={'video_sha256':meta['video_sha256'],'fps':meta['fps'],'frames':meta['frames'],
            'accepted':False,'status':'diagnostic_preview' if failures else 'awaiting_visual_review',
            'grip_from_butt_m':.09,'manual_acceptance_inherited':False,
            'motion':m,'direction':d,'contact_proxy':{k:v for k,v in contacts.items() if k!='frames'},
            'failed_numeric_gates':failures,'roles':roles,
            'candidate_sha256':sha256(result/'racket_poses.json'),
            'accuracy_validation':'automatic observations also used in fitting; not independent ground truth'}
    write(stage/'application_report.json',report)
    # Restore the untouched display metadata after real-only racket fitting.
    write(result/'mesh_meta.json',meta)
    return report


def publish(stage, dest, clip, entries, report):
    """Add an explicitly provisional preview; keep body-only viewer and geometry."""
    src=stage/'result';result=dest/'result'
    identity(json.loads((result/'mesh_meta.json').read_text()),result/'video.mp4')
    if sha256(src/'mesh_local.bin')!=sha256(result/'mesh_local.bin'):
        raise ValueError('Displayed native body differs from fitting input')
    protected={n:sha256(result/n) for n in ['mesh_meta.json','mesh_local.bin','mesh_refined.bin','video.mp4','body_texture_rgba.png']}
    candidates=['racket_poses.json','racket_poses_directional.json','racket_poses_wrist.json',
                'racket_candidates.json','racket_roi_candidates.json','racket_keypoints.json',
                'racket_grip_settings.json','racket_grip_calibration.json','racket_dimensions.json',
                'racket_quality_gate.json','wilson_mesh.bin','wilson_model.json','wilson_grasp_calibration.json',
                'grip_fit_manifest.json','grip_contact_report.json']
    for n in candidates:
        if (src/n).exists():shutil.copy2(src/n,result/n)
    write(result/'grip_application_report.json',report)
    shutil.copy2(stage/'automatic_distance/report.json',result/'automatic_grip_report.json')
    appearance={'version':1,'video_sha256':report['video_sha256'],
                'vertices':json.loads((result/'mesh_meta.json').read_text())['vertices'],
                'clips':9,'status':'shared_offline_texture_unaccepted',
                'files':{n:sha256(result/n) for n in ['appearance_map.bin','appearance_uv.bin','appearance_indices.bin','body_texture_rgba.png']}}
    write(result/'appearance_manifest.json',appearance)
    html=(MODULES/'dataset.html').read_text()
    title=f'{clip} · 人体＋球拍 · 9cm 握持候选'
    html=html.replace('新视频 · SAM 人体',title).replace('<h1>新视频 · 三维人体</h1>',f'<h1>{title}</h1>')
    current=next(i for i,e in enumerate(entries) if e['name']==clip)
    url=lambda e:e['viewer'] if e['name']=='48.43' else e['viewer'].replace('viewer.html','grip_viewer.html')
    options=''.join(f'<option value="{url(e)}"'+(' selected' if i==current else '')+f'>{i+1}/9 · {e["name"]}</option>' for i,e in enumerate(entries))
    links='<nav><a href="viewer.html">人体纹理对照</a>'
    if current:links+=f'<a href="{url(entries[current-1])}">上一段</a>'
    links+=f'<select aria-label="选择挥拍视频" onchange="location.href=this.value">{options}</select>'
    if current+1<len(entries):links+=f'<a href="{url(entries[current+1])}">下一段</a>'
    links+='<a href="/datasets/85ade7a072984579831f5cb76e8e5fd3/result/viewer.html">48.43 已验收参考</a></nav>'
    note='<p role="status">固定 9cm · 本段候选待验收 · 真人观测拟合，镜中仅供诊断 · <a href="grip_application_report.json">拟合指标</a> · <a href="automatic_grip_report.json">独立自动测距诊断</a></p>'
    html=html.replace(f'<h1>{title}</h1>',f'<h1>{title}</h1>'+links+note)
    html=html.replace('value="smooth" selected','value="smooth" disabled').replace('value="raw">','value="raw" selected>')
    html=html.replace('<script src="/assets/dataset_racket_review.js"></script>','')
    html=html.replace('id="markGround"','id="markGround" hidden').replace('id="markMirror"','id="markMirror" hidden')
    html=html.replace('id="applyMirror"','id="applyMirror" hidden').replace('id="exportMarks"','id="exportMarks" hidden')
    html=html.replace('<label class="file-label">↑ 导入标记','<label class="file-label" hidden>↑ 导入标记')
    (result/'grip_viewer.html').write_text(html)
    original=result/'viewer.html';body=original.read_text()
    if 'href="grip_viewer.html"' not in body:
        backup=stage/'original_collection_viewer.html';shutil.copy2(original,backup)
        original.write_text(body.replace('<a href="native_viewer.html">','<a href="grip_viewer.html">9cm 人体＋球拍候选</a><a href="native_viewer.html">'))
    if any(sha256(result/n)!=h for n,h in protected.items()):raise ValueError('Body/reference changed while publishing')
    write(stage/'publication.json',{'viewer':f'/datasets/{dest.name}/result/grip_viewer.html',
          'accepted':False,'protected_files':protected,'body_files_unchanged':True,
          'files':{n:sha256(result/n) for n in candidates if (result/n).exists()}})


def run(args):
    reference=args.library/'85ade7a072984579831f5cb76e8e5fd3/result/manual_grip_face_acceptance.json'
    accepted=json.loads(reference.read_text())
    if not all(accepted['accepted_scopes'].values()):raise ValueError('Reference review not accepted')
    for n,h in accepted['bound_files'].items():
        if sha256(reference.parent/n)!=h:raise ValueError('Accepted reference version changed')
    collection=json.loads(args.collection.read_text())
    entries=copy.deepcopy(collection['clips'])
    entries[0]['viewer']='/datasets/85ade7a072984579831f5cb76e8e5fd3/result/viewer.html'
    args.output.mkdir(parents=True,exist_ok=True)
    protocol={'created_at':now(),'reference_acceptance_sha256':sha256(reference),
              'grip_from_butt_m':.09,'source':'user_specified','body_refit':False,
              'manual_acceptance_inherited':False,'mirror_racket_loss_enabled':False,
              'device':args.device,'detector_sha256':sha256(args.model),'git_commit':revision(ROOT),
              'code_sha256':sha256(Path(__file__)),'entries':entries}
    write(args.output/'protocol.json',protocol)
    for entry in entries[1:]:
        name=entry['name']
        if args.clips and name not in args.clips:continue
        stage=args.output/'clips'/name
        if (stage/'publication.json').exists():
            print(name+' already completed',flush=True);continue
        if stage.exists():raise ValueError('Incomplete stage exists; inspect logs before resume: '+str(stage))
        print(name+' starting',flush=True)
        meta=prepare(args.sources/name,stage,reference)
        write(stage/'state.json',{'status':'running','stage':'enrichment','started_at':now()})
        command(stage,'enrich_result.py',['--dataset',stage,'--model',args.model,'--device',args.device])
        write(stage/'state.json',{'status':'running','stage':'original_resolution_observations','at':now()})
        command(stage,'racket_observations.py',['--dataset',stage,'--model',args.model,'--device',args.device])
        write(stage/'state.json',{'status':'running','stage':'independent_automatic_distance','at':now()})
        auto_distance(stage,args.model,args.device)
        roles=filter_roles(stage)
        write(stage/'state.json',{'status':'running','stage':'fixed_9cm_fit','at':now()})
        command(stage,'racket_keypoints.py',['--dataset',stage])
        command(stage,'fit_dataset_grip.py',['--dataset',stage])
        report=audit(stage,roles)
        publish(stage,args.library/entry['id'],name,entries,report)
        write(stage/'state.json',{'status':'completed','at':now(),'accepted':False,'numeric_failures':report['failed_numeric_gates']})
        print(json.dumps({'clip':name,'summary':json.loads((stage/'result/racket_poses.json').read_text())['summary'],
                          'failed_numeric_gates':report['failed_numeric_gates']},ensure_ascii=False),flush=True)
    if any(sha256(reference.parent/n)!=h for n,h in accepted['bound_files'].items()):
        raise ValueError('Accepted 48.43 changed during batch')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--device',default='mps')
    p.add_argument('--clips',nargs='*')
    run(p.parse_args())
