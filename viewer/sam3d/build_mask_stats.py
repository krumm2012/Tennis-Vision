"""Regenerate mask coverage statistics whenever the SAM2 atlas changes."""
from pathlib import Path
import cv2,json
p=Path(__file__).parent
a=cv2.imread(str(p/'person_masks_sam2.png'))
assert a is not None and a.shape[:2]==(5760,10240)
count=json.loads((p/'mesh_meta.json').read_text())['frames']
rows=[]
for i in range(count):
 tile=a[i//16*360:(i//16+1)*360,i%16*640:(i%16+1)*640]
 rows.append({'real':float((tile[:,:,2]>140.25).mean()),'mirror':float((tile[:,:,1]>140.25).mean())})
(p/'person_masks_sam2_stats.json').write_text(json.dumps(rows))
