import http.client, json, subprocess, tempfile, threading, time, unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from server import Library, Handler, ThreadingHTTPServer
from package_result import package

class ImportTests(unittest.TestCase):
 def setUp(self):
  self.space=patch('server.shutil.disk_usage',return_value=type('Disk',(),{'free':10**12})());self.space.start();self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.video=self.root/'clip.mp4'
  subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=64x48:r=25','-t','0.12','-pix_fmt','yuv420p',str(self.video)],check=True)
  self.library=Library(self.root/'library',[]);self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.library=self.library;self.server.origins={'http://127.0.0.1:18766'};threading.Thread(target=self.server.serve_forever,daemon=True).start()
 def tearDown(self):
  self.space.stop();self.server.shutdown();self.server.server_close();self.library.pool.shutdown();self.temp.cleanup()
 def request(self,method,path,body=None,headers=None):
  conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port);conn.request(method,path,body=body,headers=headers or {});r=conn.getresponse();status=r.status;data=r.read();conn.close();return status,data
 def test_default_viewer_and_video_are_served_without_docker(self):
  status,page=self.request('GET','/default/viewer.html');self.assertEqual(status,200);self.assertIn(b'SAM',page)
  status,part=self.request('GET','/default/video.mp4',headers={'Range':'bytes=0-15'});self.assertEqual(status,206);self.assertEqual(len(part),16)
  status,_=self.request('GET','/default/vendor/three-0.180.0.min.js');self.assertEqual(status,200)
  status,_=self.request('GET','/default/../../README.md');self.assertEqual(status,404)
  status,page=self.request('GET','/import.html');self.assertEqual(status,200);self.assertIn(b'href="/default/viewer.html"',page)
 def test_upload_isolation_range_and_unconfigured_generation(self):
  status,body=self.request('POST','/api/videos?name=clip.mp4',self.video.read_bytes());self.assertEqual(status,201);item=json.loads(body)
  self.assertEqual((self.library.folder(item['id'])/'original.video').read_bytes(),self.video.read_bytes())
  status,_=self.request('GET','/datasets/'+item['id']+'/original.video');self.assertEqual(status,404)
  status,part=self.request('GET',item['preview'],headers={'Range':'bytes=0-15'});self.assertEqual(status,206);self.assertEqual(len(part),16)
  status,_=self.request('POST','/api/videos/'+item['id']+'/generate');self.assertEqual(status,400);self.assertEqual(self.library.read(item['id'])['status'],'imported')
  status,_=self.request('POST','/api/videos?name=bad','not a video');self.assertEqual(status,400);self.assertEqual(len(self.library.list()),1)
  status,_=self.request('GET','/api/videos',headers={'Origin':'https://evil.example'});self.assertEqual(status,403)
  status,_=self.request('GET','/datasets/../../etc/passwd');self.assertEqual(status,404)
 def test_package_uses_own_geometry_and_rejects_frame_mismatch(self):
  archive=self.root/'reconstruction.npz';v=np.array([[[0,0,1],[1,0,1],[0,1,1]]]*3,dtype=np.float32)
  np.savez(archive,vertices=v,faces=np.array([[0,1,2]]),source_roots=np.zeros((3,3)),focal=np.ones(3)*50,masks=np.ones((3,24,32),np.uint8)*255)
  package(self.video,archive,self.root/'result');meta=json.loads((self.root/'result/mesh_meta.json').read_text());self.assertEqual(meta['frames'],3);self.assertEqual(meta['image_size'],[64,48]);self.assertFalse(meta['mirror_available']);self.assertTrue(meta['stabilization_available']);self.assertTrue((self.root/'result/quality_report.json').is_file())
  import cv2
  self.assertEqual(int(cv2.imread(str(self.root/'result/person_masks_sam2.png'))[0,0,2]),255)
  np.savez(archive,vertices=v[:2],faces=np.array([[0,1,2]]),source_roots=np.zeros((2,3)),focal=np.ones(2)*50,masks=np.ones((2,24,32),np.uint8))
  with self.assertRaises(ValueError):package(self.video,archive,self.root/'bad')
 def test_successful_job_publishes_own_viewer(self):
  import sys
  adapter=self.root/'adapter.py'
  adapter.write_text("import argparse,numpy as np\np=argparse.ArgumentParser();p.add_argument('--video');p.add_argument('--output');a=p.parse_args()\nnp.savez(a.output+'/reconstruction.npz',vertices=np.array([[[0,0,1],[1,0,1],[0,1,1]]]*3),faces=np.array([[0,1,2]]),source_roots=np.zeros((3,3)),focal=np.ones(3)*50,masks=np.ones((3,24,32),np.uint8)*255)\n")
  status,body=self.request('POST','/api/videos?name=generated.mp4',self.video.read_bytes());item=json.loads(body)
  self.library.command=[sys.executable,str(adapter)];self.library.generate(item['id'])
  for _ in range(150):
   state=self.library.read(item['id'])
   if state['status'] in ('ready','failed'):break
   time.sleep(.02)
  self.assertEqual(state['status'],'ready',state)
  folder=self.library.folder(item['id']);manifest=json.loads((folder/'attempts/0001/run_manifest.json').read_text());self.assertEqual(manifest['status'],'ready');self.assertEqual(manifest['source_sha256'],json.loads((folder/'result/mesh_meta.json').read_text())['video_sha256'])
  status,page=self.request('GET',state['viewer']);self.assertEqual(status,200);self.assertIn(b'SamMeshRenderer',page)
 def test_annotations_are_video_scoped_and_invalid_points_do_not_overwrite(self):
  status,body=self.request('POST','/api/videos?name=markers.mp4',self.video.read_bytes());item=json.loads(body);folder=self.library.folder(item['id']);out=folder/'result';out.mkdir()
  meta={'video_sha256':'this-video','image_size':[64,48],'frames':3,'mirror_available':True};(out/'mesh_meta.json').write_text(json.dumps(meta));self.library.update(item['id'],status='ready')
  value={'video_sha256':'other','image_size':[64,48],'ground':{'points':[[1,1],[63,1],[63,47],[1,47]],'frame':0},'mirror':{'points':[]}}
  endpoint='/api/videos/'+item['id']+'/annotations'
  status,_=self.request('POST',endpoint,json.dumps(value));self.assertEqual(status,400)
  value['video_sha256']='this-video';status,_=self.request('POST',endpoint,json.dumps(value));self.assertEqual(status,200)
  self.assertFalse(json.loads((out/'mesh_meta.json').read_text())['mirror_available'])
  value['ground']['points']=[[0,0],[63,47],[63,0],[0,47]];status,_=self.request('POST',endpoint,json.dumps(value));self.assertEqual(status,400)
  self.assertEqual(json.loads((out/'calibration_annotations.json').read_text())['ground']['points'][0],[1,1])
 def test_generation_failure_and_restart_recovery(self):
  status,body=self.request('POST','/api/videos?name=clip.mp4',self.video.read_bytes());item=json.loads(body)
  self.library.command=['/usr/bin/false'];self.library.generate(item['id'])
  for _ in range(100):
   if self.library.read(item['id'])['status']=='failed':break
   time.sleep(.02)
  self.assertEqual(self.library.read(item['id'])['status'],'failed')
  folder=self.library.folder(item['id']);path=folder/'attempts/0001/run_manifest.json';self.assertEqual(json.loads(path.read_text())['status'],'failed')
  self.library.update(item['id'],status='generating');other=Library(self.library.root,[]);self.assertEqual(other.read(item['id'])['status'],'failed');self.assertEqual(json.loads(path.read_text())['error_type'],'ServiceRestart');other.pool.shutdown()

if __name__=='__main__':unittest.main()
