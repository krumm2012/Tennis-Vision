"""Import all frozen clips as a browsable, separate texture review collection."""
import argparse
import json
import uuid
import re
from pathlib import Path
from publish_texture_review import publish
from run_records import write, sha256


def publish_collection(batch, fusion, baseline, layout, library):
    clips = json.loads((batch / 'batch_manifest.json').read_text())['clips']
    report = json.loads((fusion / 'texture_report.json').read_text())
    rows = {row['video']: row for row in report['validation']}
    entries = []
    for clip in clips:
        name = Path(clip['folder']).name
        ident = uuid.uuid5(uuid.NAMESPACE_URL, 'tennis-vision:texture-collection-v1:' + name).hex
        row = rows[name]
        entries.append({'id': ident, 'name': name, 'viewer': f'/datasets/{ident}/result/viewer.html',
                        'train_face_coverage': row['train_face_coverage'],
                        'video_sha256': sha256(Path(clip['folder']) / 'source.mp4')})
    collection = {'schema_version': 1, 'accepted': False, 'clips': entries,
                  'shared_atlas_uv_coverage': report['atlas_observed_texel_fraction'],
                  'shared_face_coverage': report['combined_face_coverage'],
                  'texture_sha256': sha256(fusion / 'body_texture_rgba.png'),
                  'scope': 'native mesh playback + shared offline atlas, no joint body/racket refinement'}
    for entry in entries:
        dest = library / entry['id']
        publish(batch, fusion, baseline, layout, dest, entry['name'])
        source = next(Path(c['folder']) / 'result' for c in clips if Path(c['folder']).name == entry['name'])
        native = dest / 'result'
        native.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name != 'viewer.html':
                target = native / path.name
                if not target.exists():
                    target.hardlink_to(path.resolve())
                elif sha256(target) != sha256(path):
                    raise ValueError('Native comparison source changed')
        html = (source / 'viewer.html').read_text()
        html = html.replace('<h1>新视频 · 三维人体</h1>', f'<h1>{entry["name"]} · 原生单次输出对照</h1><p><a href="viewer.html">返回九段遍历与共享纹理</a> · 原始姿态 / 当前帧投影贴图；未启用稳定姿态或跨帧补色。遮罩与可见性判断仍参与显示。</p>')
        html = html.replace('value="smooth" selected', 'value="smooth" disabled').replace('value="raw">', 'value="raw" selected>')
        html = html.replace('renderer=new SamMeshRenderer(canvas,video);', 'renderer=new SamMeshRenderer(canvas,video);renderer.useTemporalTexture=false;renderer.edgeRepair=false;')
        # The original template advertises racket tooling, but these clips have no fitted racket.
        html = html.replace('SAM 人体 · Wilson 球拍 · 配对镜面标记', '原生 SAM 人体 · 当前帧贴图覆盖率见页面下方')
        html = html.replace('id="gripFocus"', 'id="gripFocus" hidden')
        html = re.sub(r'<details id="datasetTools">.*?</details>', '<details open><summary>当前视角贴图覆盖率</summary><p id="textureStatus" role="status"></p><input id="mirrorEnabled" type="checkbox" hidden><p>按当前相机可见像素统计；与共享 UV 像素覆盖率的分母不同。未使用镜中补色。</p></details>', html)
        html = re.sub(r'<script src="/assets/(?:dataset_[^"]+|coaching)\.js"></script>', '', html)
        html = re.sub(r'<script>TennisCoaching\.mount.*?</script>', '', html)
        (native / 'native_viewer.html').write_text(html)
        write(dest / 'result/collection.json', collection)
        record = json.loads((dest / 'record.json').read_text())
        record.update(name=f"九段遍历 · {entry['name']} · 纹理候选",
                      message=f"原生人体 · 单段面覆盖 {entry['train_face_coverage']:.2%} · 共享 UV {collection['shared_atlas_uv_coverage']:.2%} · 未验收",
                      texture_collection=True, source_sha256=entry['video_sha256'])
        write(dest / 'record.json', record)
    return collection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'fusion', 'baseline', 'layout', 'library'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = publish_collection(args.batch, args.fusion, args.baseline, args.layout, args.library)
    print(json.dumps(result, ensure_ascii=False, indent=2))
