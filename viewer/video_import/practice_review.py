"""Video-bound, revisioned coach reviews. No pose or racket files are modified."""
from copy import deepcopy
import json
import re
from pathlib import Path
from practice_scoring import POLICY_VERSION, POLICY_SHA256, score_review
from run_records import sha256, now, write


def bindings(result):
    names = ['mesh_meta.json', 'racket_poses.json', 'mesh_local.bin', 'mesh_refined.bin', 'mesh_smooth.bin']
    return {name: sha256(result/name) for name in names if (result/name).is_file()}


def load(result):
    meta = json.loads((result/'mesh_meta.json').read_text())
    path = result/'practice_review.json'
    document = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 1, 'policy_version': POLICY_VERSION, 'policy_sha256': POLICY_SHA256,
        'video_sha256': meta['video_sha256'], 'revision': 0, 'shots': []}
    training_path=result/'machine_training_context.json'
    training=json.loads(training_path.read_text()) if training_path.exists() else None
    if training and training.get('video_sha256')!=meta['video_sha256']:
        training=None
    return {'training_context': training, 'document': document, 'evidence_sha256': bindings(result),
            'policy_sha256': POLICY_SHA256, 'video_sha256': meta['video_sha256'], 'frames': meta['frames'], 'fps': meta['fps']}


def text(value, name, required=False, maximum=1000):
    if not isinstance(value, str) or len(value) > maximum or (required and not value.strip()):
        raise ValueError(f'{name}无效')
    return value.strip()


def save(result, payload):
    current = load(result)
    document = current['document']
    if payload.get('video_sha256') != current['video_sha256']:
        raise ValueError('评价属于其他视频')
    if payload.get('policy_version') != POLICY_VERSION or payload.get('policy_sha256') != POLICY_SHA256:
        raise ValueError('评分标准已更新，请刷新')
    if type(payload.get('base_revision')) is not int or payload['base_revision'] != document['revision']:
        raise ValueError('评价已被更新，请刷新后继续，避免覆盖他人记录')
    if payload.get('evidence_sha256') != current['evidence_sha256']:
        raise ValueError('动作重建已变化，请刷新并重新复核')
    if not isinstance(payload.get('shot'), dict):
        raise ValueError('缺少动作评价')
    row = deepcopy(payload['shot'])
    if not isinstance(row.get('id'), str) or not re.fullmatch('[0-9a-f]{32}', row['id']):
        raise ValueError('评价编号无效')
    for key in ['start_frame', 'end_frame']:
        if type(row.get(key)) is not int:
            raise ValueError('动作帧号必须为整数')
    if not 0 <= row['start_frame'] < row['end_frame'] < current['frames']:
        raise ValueError('动作区间超出视频或顺序错误')
    if row.get('stroke') not in ['Forehand', 'Backhand', 'Two-Handed Backhand', 'Volley', 'Serve', 'Unknown']:
        raise ValueError('动作类型无效')
    if row.get('pose_variant') not in ['raw', 'smooth', 'refined']:
        raise ValueError('姿态版本无效')
    row['goal'] = text(row.get('goal'), '练习目标', maximum=200)
    row['reviewer'] = text(row.get('reviewer'), '教练姓名', maximum=80)
    row['observation'] = text(row.get('observation'), '评价依据')
    if type(row.get('confirmed')) is not bool:
        raise ValueError('确认状态无效')
    if row['confirmed'] and (not row['goal'] or not row['reviewer'] or not row['observation'] or row['stroke'] == 'Unknown'):
        raise ValueError('确认评分需填写教练、练习目标、动作类型和评价依据')
    context = row.get('machine') or {}
    if not isinstance(context, dict) or set(context) - {'speed', 'frequency', 'direction', 'spin'}:
        raise ValueError('发球机条件无效')
    row['machine'] = {k: text(context.get(k, ''), '发球机条件', maximum=100)
                      for k in ['speed', 'frequency', 'direction', 'spin']}
    row['score'] = score_review(row)
    if row['confirmed'] and row['score']['score'] is None:
        raise ValueError('五项全部评分后才能确认')
    row['updated_at'] = now()
    row['evidence_sha256'] = current['evidence_sha256']
    row['video_sha256'] = current['video_sha256']
    row['comparison_ready'] = bool(row['goal'] and all(row['machine'].values()))
    updated = deepcopy(document)
    updated['shots'] = [s for s in document['shots'] if s['id'] != row['id']] + [row]
    if len(updated['shots']) > 200:
        raise ValueError('单视频评价数量超过限制')
    updated.update(policy_version=POLICY_VERSION, policy_sha256=POLICY_SHA256,
                   revision=document['revision']+1, updated_at=now())
    history = result/f"practice_review_revision_{updated['revision']:06d}.json"
    # History is append-only. The handler serializes saves under the library lock.
    with history.open('x') as stream:
        json.dump(updated, stream, ensure_ascii=False, indent=2)
    write(result/'practice_review.json', updated)
    return {'document': updated, 'evidence_sha256': current['evidence_sha256']}
