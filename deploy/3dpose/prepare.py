"""Package only the local data needed to serve the SAM 3D Viewer."""
from pathlib import Path
import hashlib
import gzip
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'viewer' / 'sam3d'
DATA = ROOT / 'output' / 'sam3d_cloud'
TARGET = ROOT / 'output' / '3dpose_docker'
FILES = (
    'video.mp4', 'temporal_pose.json', 'mesh_meta.json',
    'mesh_local.bin', 'mesh_smooth.bin', 'mesh_refined.bin',
    'mesh_temporal.bin', 'mesh_faces.bin',
    'mirror_geometry.json', 'mirror_geometry_frames.json',
    'person_masks_sam2.png', 'person_masks_sam2_stats.json',
    'temporal_texture_sam2.bin', 'mirror_ground_grid.json',
    'mirror_corner_suggestion.json', 'paired_ground_diagnostic.json',
)
PAGE = ('viewer.html', 'mesh_renderer.js', 'texture_audit.js',
        'calibration_editor.js', 'mirror_grid_editor.js', 'racket_editor.js')


def prepare() -> Path:
    TARGET.mkdir(parents=True, exist_ok=True)
    public = TARGET / 'public'
    public.mkdir(exist_ok=True)
    manifest = []
    for name in PAGE + FILES:
        origin = SOURCE / name if name in PAGE else DATA / name
        if not origin.is_file():
            raise FileNotFoundError(f'Required Viewer file missing: {origin}')
        dest = public / name
        shutil.copy2(origin, dest)
        with dest.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest.append({'name': name, 'bytes': dest.stat().st_size, 'sha256': digest})
        if name in ('temporal_texture_sam2.bin', 'temporal_pose.json'):
            with dest.open('rb') as source, (public / (name + '.gz')).open('wb') as output:
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=6, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed)
    for name in ('Dockerfile', 'nginx.conf', 'compose.yaml'):
        shutil.copy2(Path(__file__).parent / name, TARGET / name)
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return TARGET


if __name__ == '__main__':
    path = prepare()
    print(f'Packaged {len(PAGE) + len(FILES)} Viewer files in {path}')
