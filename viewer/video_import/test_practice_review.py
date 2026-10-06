import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).parent))
from practice_review import load, save
from practice_scoring import DIMENSIONS, POLICY_VERSION, POLICY_SHA256
from server import Handler, Library


class PracticeReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.ident='a'*32;self.result=self.root/self.ident/'result';self.result.mkdir(parents=True)
        (self.result/'mesh_meta.json').write_text(json.dumps({'video_sha256':'b'*64,'frames':250,'fps':25}))
        (self.result/'racket_poses.json').write_text('{}')
        state=load(self.result)
        self.payload={
            'video_sha256':state['video_sha256'],'policy_version':POLICY_VERSION,'policy_sha256':POLICY_SHA256,
            'base_revision':0,'evidence_sha256':state['evidence_sha256'],
            'shot':{'id':'c'*32,'start_frame':12,'end_frame':46,'stroke':'Forehand','goal':'固定落点正手节奏',
                    'reviewer':'TEST_COACH','observation':'TEST_FIXTURE_ONLY','confirmed':True,
                    'ratings':{k:4 for k in DIMENSIONS},'machine':{},'pose_variant':'raw'}}

    def tearDown(self): self.temp.cleanup()

    def test_confirmation_save_reload_and_append_only_revisions(self):
        before=(self.result/'racket_poses.json').read_bytes()
        result=save(self.result,self.payload)
        self.assertEqual(result['document']['shots'][0]['score']['score'],80)
        self.assertFalse(result['document']['shots'][0]['comparison_ready'])
        self.assertEqual(load(self.result)['document'],result['document'])
        self.assertEqual((self.result/'racket_poses.json').read_bytes(),before)
        p=copy.deepcopy(self.payload);p['base_revision']=1;p['shot']['ratings']['preparation']=5
        save(self.result,p)
        historical=json.loads((self.result/'practice_review_revision_000001.json').read_text())
        self.assertEqual(historical['shots'][0]['score']['score'],80)
        self.assertEqual(load(self.result)['document']['shots'][0]['score']['score'],84)

    def test_stale_video_evidence_policy_and_revision_are_rejected(self):
        for key,value in [('video_sha256','d'*64),('policy_sha256','x'),('policy_version','x'),('base_revision',1),('evidence_sha256',{})]:
            p=copy.deepcopy(self.payload);p[key]=value
            with self.assertRaises(ValueError):save(self.result,p)
        (self.result/'racket_poses.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):save(self.result,self.payload)
        self.assertFalse((self.result/'practice_review.json').exists())

    def test_drafts_keep_missing_ratings_unknown(self):
        p=copy.deepcopy(self.payload);p['shot'].update(confirmed=False,reviewer='',observation='',goal='')
        p['shot']['ratings']['contact']=None
        result=save(self.result,p)
        self.assertIsNone(result['document']['shots'][0]['score']['score'])
        self.assertEqual(result['document']['shots'][0]['score']['coverage'],.8)

    def test_confirmation_requires_dimensions_and_review_context(self):
        for key,value in [('reviewer',''),('goal',''),('observation',''),('stroke','Unknown')]:
            p=copy.deepcopy(self.payload);p['shot'][key]=value
            with self.assertRaises(ValueError):save(self.result,p)
        p=copy.deepcopy(self.payload);p['shot']['ratings']['contact']=None
        with self.assertRaises(ValueError):save(self.result,p)

    def test_interval_and_rating_validation(self):
        for start,end in [(5,5),(-1,20),(20,10),(0,250),(True,20),(2.5,20)]:
            p=copy.deepcopy(self.payload);p['shot'].update(start_frame=start,end_frame=end)
            with self.assertRaises(ValueError):save(self.result,p)
        for ratings in [[], '4', 1]:
            p=copy.deepcopy(self.payload);p['shot']['ratings']=ratings
            with self.assertRaises(ValueError):save(self.result,p)
        for value in [True,0,6,2.5,'3']:
            p=copy.deepcopy(self.payload);p['shot']['ratings']['contact']=value
            with self.assertRaises(ValueError):save(self.result,p)

    def test_loopback_http_roundtrip_and_conflict(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        server.library=Library(self.root,[]);server.origins=set()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/practice-policy') as response: self.assertEqual(json.load(response)['version'],POLICY_VERSION)
            route=base+'/api/videos/'+self.ident+'/practice-review'
            with urlopen(route) as response:self.assertEqual(json.load(response)['document']['revision'],0)
            request=Request(route,data=json.dumps(self.payload).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request) as response:self.assertEqual(json.load(response)['document']['revision'],1)
            with self.assertRaises(HTTPError) as error:urlopen(request)
            self.assertEqual(error.exception.code,400)
            error.exception.close()
            with urlopen(route) as response:self.assertEqual(len(json.load(response)['document']['shots']),1)
        finally:
            server.shutdown();server.server_close();thread.join();server.library.pool.shutdown()
