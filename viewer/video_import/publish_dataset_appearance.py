"""Bind a verified same-subject UV candidate to a complete dataset Viewer.

Only videos present in the fusion input manifest are eligible. Preserve all pose,
mesh and annotation files; a manifest written last activates the appearance.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from run_records import sha256,write


def publish(result,fusion,layout):
    result,fusion,layout=map(Path,(result,fusion,layout))
    meta=json.loads((result/'mesh_meta.json').read_text());report=json.loads((fusion/'texture_report.json').read_text())
    if meta['video_sha256'] not in {r['video_sha256'] for r in report['inputs']}:raise ValueError('Video is not part of this appearance capture')
    if report['layout_sha256']!=sha256(layout) or report['artifact_sha256']['body_texture_rgba.png']!=sha256(fusion/'body_texture_rgba.png'):raise ValueError('Appearance provenance mismatch')
    with np.load(layout,allow_pickle=False) as rig:
        faces=rig['faces'];uv_faces=rig['uv_faces'];uv=rig['uv'];mapping=np.full(len(uv),-1,np.int32)
        if not np.array_equal(np.fromfile(result/'mesh_faces.bin',dtype='<u4'),faces.ravel()):raise ValueError('Body topology mismatch')
        if faces.max()>=meta['vertices']:raise ValueError('Body vertex count mismatch')
        for ids,vertices in zip(uv_faces,faces):
            for i,v in zip(ids,vertices):
                if mapping[i]>=0 and mapping[i]!=v:raise ValueError('Conflicting UV correspondence')
                mapping[i]=v
        mapping[mapping<0]=0
        arrays={'appearance_map.bin':mapping.astype('<u4'),'appearance_uv.bin':uv.astype('<f4'),'appearance_indices.bin':uv_faces.astype('<u4')}
    # Back up a previous binding before replacing any of its assets.
    backup=result/'appearance_previous'
    names=[*arrays,'body_texture_rgba.png','appearance_manifest.json','viewer.html']
    for name in names:
        if (result/name).exists():
            backup.mkdir(exist_ok=True);shutil.copy2(result/name,backup/name)
    for name,array in arrays.items():
        pending=result/(name+'.pending');array.tofile(pending);pending.replace(result/name)
    pending=result/'body_texture_rgba.pending.png';shutil.copy2(fusion/'body_texture_rgba.png',pending);pending.replace(result/'body_texture_rgba.png')
    manifest={'version':1,'video_sha256':meta['video_sha256'],'vertices':meta['vertices'],'clips':report['clips'],'status':report['status'],'layout_sha256':sha256(layout),'source_report_sha256':sha256(fusion/'texture_report.json'),'files':{name:sha256(result/name) for name in [*arrays,'body_texture_rgba.png']}}
    write(result/'appearance_manifest.json',manifest)
    shutil.copy2(Path(__file__).with_name('dataset.html'),result/'viewer.html')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['result','fusion','layout']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(publish(a.result,a.fusion,a.layout),indent=2))
