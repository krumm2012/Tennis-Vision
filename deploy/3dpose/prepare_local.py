"""Stage a local-only Docker UI with a read-only video library and Mac API proxy."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'viewer/video_import'))
from run_records import revision,sha256,write

def main():
    target=ROOT/'output/3dpose_local';public=target/'public';public.mkdir(parents=True,exist_ok=True)
    assets={
      'import.html':ROOT/'viewer/video_import/import.html',
      'assets/dataset_appearance.js':ROOT/'viewer/video_import/dataset_appearance.js',
      'assets/dataset_racket.js':ROOT/'viewer/video_import/dataset_racket.js',
      'assets/dataset_racket_review.js':ROOT/'viewer/video_import/dataset_racket_review.js',
      'assets/dataset_tools.js':ROOT/'viewer/video_import/dataset_tools.js',
      'assets/mesh_renderer.js':ROOT/'viewer/sam3d/mesh_renderer.js',
      'assets/practice_coach.js':ROOT/'viewer/sam3d/practice_coach.js',
      'assets/coaching.js':ROOT/'viewer/sam3d/coaching.js',
      'assets/vendor/three-0.180.0.min.js':ROOT/'viewer/sam3d/vendor/three-0.180.0.min.js'}
    manifest=[]
    for name,source in assets.items():
        dest=public/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);manifest.append({'path':name,'sha256':sha256(dest)})
    for origin,dest in [('Dockerfile.local','Dockerfile'),('nginx.local.conf','nginx.conf'),('compose.local.yaml','compose.yaml')]:shutil.copy2(Path(__file__).parent/origin,target/dest)
    library=ROOT/'output/video_library';library.mkdir(parents=True,exist_ok=True)
    write(public/'release_manifest.json',{'code_revision':revision(ROOT),'scope':'local_only','assets':manifest,'datasets':'read_only_live_mount','api':'Mac_loopback_18768'})
    (target/'.env').write_text('CODE_REVISION='+revision(ROOT)+'\nVIDEO_LIBRARY_PATH='+json.dumps(str(library))+'\n')
    print(target)

if __name__=='__main__':main()
