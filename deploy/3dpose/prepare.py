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
        'calibration_editor.js', 'mirror_grid_editor.js', 'racket_editor.js',
        'racket_renderer.js')


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
    def package(origin: Path, relative: str) -> None:
        if not origin.is_file():
            raise FileNotFoundError(f'Required release asset missing: {origin}')
        dest = public / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Follow local links: the container must not depend on a Mac filesystem path.
        shutil.copy2(origin, dest, follow_symlinks=True)
        with dest.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest.append({'name': relative, 'bytes': dest.stat().st_size, 'sha256': digest})
        if dest.name in ('temporal_texture_sam2.bin', 'temporal_pose.json', 'viewer_data.json'):
            with dest.open('rb') as source, Path(str(dest) + '.gz').open('wb') as output:
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=6, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed)

    for name in ('racket_poses.json', 'wilson_mesh.bin', 'wilson_model.json',
                 'racket_poses_v1.json', 'racket_poses_v2.json', 'racket_poses_v3.json'):
        package(DATA / name, name)
    optional = DATA / 'racket_annotations.json'
    if optional.is_file():
        package(optional, optional.name)
    else:
        (public / optional.name).unlink(missing_ok=True)
    joint = DATA / 'joint_fit_v4'
    for name in ('report.json', 'quality.json', 'viewer_data.json'):
        package(joint / name, 'joint_fit_v4/' + name)
    package(SOURCE / 'joint_fit/viewer.html', 'joint_fit_v4/viewer.html')
    full = joint / 'full'
    for name in PAGE + FILES + ('ground_calibration.json', 'racket_poses.json', 'wilson_mesh.bin'):
        origin = full / name
        if name in PAGE and name != 'viewer.html':
            origin = SOURCE / name
        elif not origin.is_file() and name in FILES:
            origin = DATA / name
        package(origin, 'joint_fit_v4/full/' + name)
    for asset in sorted((SOURCE / 'vendor').iterdir()):
        if asset.is_file():
            package(asset, 'vendor/' + asset.name)
            package(asset, 'joint_fit_v4/full/vendor/' + asset.name)
    # Expose the exact content inventory for deployment verification.
    (public / 'release_manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in ('Dockerfile', 'nginx.conf', 'compose.yaml'):
        shutil.copy2(Path(__file__).parent / name, TARGET / name)
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return TARGET


if __name__ == '__main__':
    path = prepare()
    print(f'Packaged {len(json.loads((path / "manifest.json").read_text()))} Viewer assets in {path}')
