"""Publish a separate localhost texture review; preserve accepted datasets."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from run_records import write,sha256

def publish(batch,fusion,baseline,layout,destination,clip=None):
    report=json.loads((fusion/'texture_report.json').read_text())
    clips=json.loads((batch/'batch_manifest.json').read_text())['clips']
    selected=clips[0] if clip is None else next(c for c in clips if Path(c['folder']).name==clip)
    first=Path(selected['folder'])
    source_meta=json.loads((first/'result/mesh_meta.json').read_text())
    if source_meta['video_sha256']!=sha256(first/'source.mp4'):raise ValueError('Clip mesh/video provenance mismatch')
    baseline_report=json.loads((baseline/'texture_report.json').read_text())
    if (destination/'record.json').exists() and not json.loads((destination/'record.json').read_text()).get('texture_review'):raise ValueError('Cannot replace an accepted dataset with a texture review')
    if report['layout_sha256']!=sha256(layout) or report['artifact_sha256']['body_texture_rgba.png']!=sha256(fusion/'body_texture_rgba.png'):raise ValueError('Texture candidate provenance mismatch')
    if baseline_report['layout_sha256']!=sha256(layout) or baseline_report['artifact_sha256']['body_texture_rgba.png']!=sha256(baseline/'body_texture_rgba.png'):raise ValueError('Texture baseline provenance mismatch')
    destination.mkdir(exist_ok=True,parents=True);(destination/'result').mkdir(exist_ok=True)
    out=destination/'result'
    with np.load(layout,allow_pickle=False) as rig:
        faces=rig['faces'];uv_faces=rig['uv_faces'];mapping=np.full(len(rig['uv']),-1,np.int32)
        for ids,vertices in zip(uv_faces,faces):
            for i,v in zip(ids,vertices):
                if mapping[i]>=0 and mapping[i]!=v:raise ValueError('UV vertex has conflicting mesh correspondence')
                mapping[i]=v
        # Unused UV coordinates never appear in indices; map them safely to zero.
        mapping[mapping<0]=0;mapping.astype('<u4').tofile(out/'appearance_map.bin');rig['uv'].astype('<f4').tofile(out/'appearance_uv.bin');uv_faces.astype('<u4').tofile(out/'appearance_indices.bin')
    for name in ['body_texture_rgba.png','texture_report.json']:shutil.copy2(fusion/name,out/name)
    shutil.copy2(baseline/'body_texture_rgba.png',out/'body_texture_baseline.png');shutil.copy2(first/'result/mesh_meta.json',out/'mesh_meta.json')
    shutil.copy2(baseline/'texture_report.json',out/'texture_baseline_report.json')
    for source,name in [(first/'result/mesh_local.bin','mesh_local.bin'),(first/'source.mp4','video.mp4')]:
        if not (out/name).exists():(out/name).hardlink_to(source.resolve())
        elif sha256(out/name)!=sha256(source):raise ValueError('Review source changed')
    shutil.copy2(Path(__file__).with_name('texture_preview.html'),out/'viewer.html')
    meta=json.loads((out/'mesh_meta.json').read_text());write(destination/'record.json',{'id':destination.name,'name':'多视频人体纹理 · 待复核','created':time.time(),'status':'ready','message':'独立 UV 合成候选，未替换已验收 Viewer','duration':meta['frames']/meta['fps'],'viewer':f'/datasets/{destination.name}/result/viewer.html','preview':f'/datasets/{destination.name}/result/video.mp4','attempt':0,'texture_review':True})
    return f'http://127.0.0.1:18769/datasets/{destination.name}/result/viewer.html'

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['batch','fusion','baseline','layout','destination']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--clip',help='Clip folder name; defaults to first clip')
    a=p.parse_args();print(publish(a.batch,a.fusion,a.baseline,a.layout,a.destination,a.clip))
