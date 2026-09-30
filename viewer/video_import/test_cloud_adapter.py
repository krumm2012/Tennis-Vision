import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cloud_adapter as a
from run_records import sha256

class AdapterTests(unittest.TestCase):
 def test_connection_config_rejects_shell_and_invalid_ports(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'config.json';c={'host':'root@gpu.example','port':33966,'root':'/root/tennis','python':'/venv/bin/python'};p.write_text(json.dumps(c));self.assertEqual(a.config(p)['port'],33966)
   for field,value in [('host','-oProxyCommand=bad'),('root','/tmp/../root'),('port',0),('python','/bin/python;touch')]:
    p.write_text(json.dumps({**c,field:value}))
    with self.assertRaises(ValueError):a.config(p)
 def transport(self,corrupt=False,failed=False):
  tmp=tempfile.TemporaryDirectory();root=Path(tmp.name);video=root/'video.mp4';video.write_bytes(b'video');out=root/'out'
  manifest={'source_sha256':sha256(video),'artifact_sha256':{'reconstruction.npz':'bad' if corrupt else __import__('hashlib').sha256(b'archive').hexdigest()}}
  def ssh(c,args):return json.dumps({'status':'failed' if failed else 'ready'}) if 'status' in args else '{}'
  def copy(c,source,dest,upload=False):
   if upload:return
   name=Path(source).name
   Path(dest).write_bytes(b'archive' if name=='reconstruction.npz' else json.dumps(manifest).encode() if name=='run_manifest.json' else b'log')
  return tmp,video,out,ssh,copy
 def test_download_is_verified_before_publish(self):
  tmp,v,out,ssh,copy=self.transport()
  with tmp,patch.object(a,'ssh',ssh),patch.object(a,'copy',copy):
   a.run(v,out,{'root':'/gpu','python':'/python'},poll=0)
   self.assertEqual((out/'reconstruction.npz').read_bytes(),b'archive');self.assertTrue((out/'worker.log').exists())
 def test_corruption_or_failure_never_publishes_archive(self):
  for corrupt,failed in [(True,False),(False,True)]:
   tmp,v,out,ssh,copy=self.transport(corrupt,failed)
   with tmp,patch.object(a,'ssh',ssh),patch.object(a,'copy',copy):
    with self.assertRaises((ValueError,RuntimeError)):a.run(v,out,{'root':'/gpu','python':'/python'},poll=0)
    self.assertFalse((out/'reconstruction.npz').exists());self.assertTrue((out/'worker.log').exists())

if __name__=='__main__':unittest.main()
