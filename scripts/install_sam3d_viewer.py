"""Install tracked Viewer sources beside locally generated SAM3D data."""
import argparse
import shutil
from pathlib import Path


def install(destination: Path) -> None:
    source = Path(__file__).resolve().parents[1] / 'viewer' / 'sam3d'
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix in {'.html', '.js', '.py'}:
            shutil.copy2(path, destination / path.name)
    shutil.copytree(source / 'vendor', destination / 'vendor', dirs_exist_ok=True)
    # Static diagnostic UI only; candidate data stays separate and is never promoted.
    diagnostic = destination / 'joint_fit_v4'
    diagnostic.mkdir(exist_ok=True)
    shutil.copy2(source / 'joint_fit' / 'viewer.html', diagnostic / 'viewer.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/sam3d_cloud'))
    install(parser.parse_args().output)
