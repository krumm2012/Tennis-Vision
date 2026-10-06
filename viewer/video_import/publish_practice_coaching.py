"""Add the practice review UI to a published collection without changing fitting evidence."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from run_records import sha256, write


def coaching_notice():
    return '<div class="notice" id="practiceCoachingNotice"><strong>教练评价已接入九段 Viewer</strong><p>进入各段页面底部的“发球机练习 · 教练评价”，圈定单次挥拍区间，再按准备与引拍、移动到位、击球位置、身体协调、随挥与回位五项评分。1–5 分标准统一，全部填写并经教练确认后显示总分；未知不补分。</p><p>ball_machine_practice_v1 · 草案待独立教练样本校准。自动 3D 教学评分暂不可用；握拍验收、测距准确性、贴图覆盖率与动作评分分别记录。</p></div>'


def publish(library, collection, output):
    summary=json.loads(collection.read_text());reference=summary['reference_acceptance']['dataset_id']
    paths=[library/reference/'result/viewer.html']
    for row in summary['clips']:
        url=urlsplit(row['viewer']).path
        parts=url.split('/')
        paths.append(library/parts[2]/'result'/parts[-1])
    output.mkdir(parents=True,exist_ok=True)
    protected={}
    for path in paths:
        result=path.parent
        names={p.name for p in result.glob('*.bin')} | {'mesh_meta.json','racket_poses.json'}
        if (result/'manual_grip_face_acceptance.json').exists():
            names.update(json.loads((result/'manual_grip_face_acceptance.json').read_text())['bound_files'])
            names.add('manual_grip_face_acceptance.json')
        protected[str(result)]={n:sha256(result/n) for n in sorted(names) if (result/n).exists()}
        backup=output/'before'/result.parent.name;backup.mkdir(parents=True,exist_ok=True)
        (backup/path.name).write_bytes(path.read_bytes())
        text=path.read_text()
        if '/assets/practice_coach.js' not in text:
            if '<script src="/assets/coaching.js"></script>' not in text:raise ValueError('Viewer has no coaching mount')
            text=text.replace('<script src="/assets/coaching.js"></script>', '<script src="/assets/practice_coach.js"></script><script src="/assets/coaching.js"></script>')
            path.write_text(text)
    dashboard=library/reference/'result/grip_collection_review.html'
    (output/'before/grip_collection_review.html').write_bytes(dashboard.read_bytes())
    text=dashboard.read_text()
    if 'id="practiceCoachingNotice"' not in text:
        dashboard.write_text(text.replace('<h2>固定9cm应用与播放检查</h2>',coaching_notice()+'<h2>固定9cm应用与播放检查</h2>'))
    for result,files in protected.items():
        for name,expected in files.items():
            if sha256(Path(result)/name)!=expected:raise ValueError('Fitting evidence modified during UI publication')
    write(output/'publication.json',{'viewers':[str(p) for p in paths],'protected_files':protected,
          'protected_files_unchanged':True,'scope':'HTML only; no grades created','policy_version':'ball_machine_practice_v1'})
    print(json.dumps({'viewers':len(paths),'protected_files':sum(len(v) for v in protected.values()),'unchanged':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();publish(a.library,a.collection,a.output)
