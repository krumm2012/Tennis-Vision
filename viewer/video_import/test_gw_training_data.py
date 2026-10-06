import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from gw_training_data import validate_page, fetch_user, timestamp, machine_fields, candidates, context

USER='15226080802'


def page(rows, index=1, total=None, next_page=False):
    return {'status':'success','data':{'user_info':{'phone_number':USER,'unified_user_id':255},
      'records':rows,'statistics':{},'pagination':{'page':index,'total_records':len(rows) if total is None else total,'has_next':next_page}}}


def snapshot(rows):
    return {'user':USER,'origin':'https://51alljoin.cn:19999','queried_at':'2026-10-03T00:00:00Z',
            'repository_reference':{},'missing_detail_fields':[],'records':rows}


class GwTrainingDataTests(unittest.TestCase):
    def test_user_identity_checked_before_accepting_records(self):
        p=page([]);p['data']['user_info']['phone_number']='10000000000'
        with self.assertRaises(ValueError):validate_page(p,USER)
        p=page([{'device_id':1,'game_id':2,'unified_user_id':999}])
        with self.assertRaises(ValueError):validate_page(p,USER)

    def test_pagination_keeps_same_game_ids_on_different_devices(self):
        pages=[page([{'device_id':1,'game_id':1}],1,2,True),page([{'device_id':2,'game_id':1}],2,2,False)]
        calls=[]
        def opener(url,timeout):
            calls.append(url);return io.BytesIO(json.dumps(pages[len(calls)-1]).encode())
        with tempfile.TemporaryDirectory() as td:
            result=fetch_user(USER,Path(td),opener)
            self.assertEqual(len(result['records']),2)
            self.assertTrue(all('/api/user/'+USER+'/records?' in url for url in calls))
            self.assertFalse(any('/sync' in url for url in calls))
            with self.assertRaises(ValueError):fetch_user(USER,Path(td),opener)

    def test_repeated_pages_and_wrong_totals_fail(self):
        rows=[{'device_id':1,'game_id':1}]
        for pages in [[page(rows,1,2,True),page(rows,2,2,False)], [page(rows,1,2,False)], [page(rows,2,1,False)]]:
            responses=iter(pages)
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(ValueError):fetch_user(USER,Path(td),lambda *a,**k:io.BytesIO(json.dumps(next(responses)).encode()))

    def test_history_months_away_does_not_fill_video_conditions(self):
        clip={'start':'2026-09-30T02:48:43Z','end':'2026-09-30T02:48:53Z'}
        rows=[{'device_id':2,'game_id':6547,'created_at':'2026-08-23 18:52:37',
               'serve_interval':2,'total_score':1085}]
        result=context(snapshot(rows),'a'*64,clip)
        self.assertEqual(result['status'],'no_matching_records')
        self.assertIsNone(result['selected_record'])
        self.assertTrue(all(v is None for v in result['machine'].values()))
        self.assertFalse(result['comparison_ready'])
        self.assertIsNone(result['movement_score'])

    def test_start_only_does_not_invent_game_duration(self):
        rows=[{'device_id':2,'game_id':1,'created_at':'2026-09-30 10:00:00','serve_interval':4,'serve_count':1000}]
        self.assertEqual(candidates(snapshot(rows),timestamp('2026-09-30T02:48:43Z'),timestamp('2026-09-30T02:48:53Z')),[])

    def test_observed_serve_timestamp_produces_candidate_not_confirmed_link(self):
        rows=[{'device_id':2,'game_id':1,'created_at':'2026-09-30 10:00:00',
               'serve_records':[{'created_at':'2026-09-30 10:48:45'}]}]
        result=context(snapshot(rows),'a'*64,{'start':'2026-09-30T02:48:43Z','end':'2026-09-30T02:48:53Z'})
        self.assertEqual(len(result['candidates']),1)
        self.assertEqual(result['candidates'][0]['observed_serves_near_video'],1)
        self.assertIsNone(result['selected_record'])
        self.assertFalse(result['comparison_ready'])

    def test_speed_setting_and_measured_ball_speed_are_not_equated(self):
        fields=machine_fields({'serve_speed':0,'ball_speed':90,'serve_interval':2,'serve_type':1})
        self.assertEqual(fields['speed']['value'],0)
        self.assertEqual(fields['speed']['unit'],'unverified_device_setting')
        self.assertFalse(fields['speed']['usable_for_comparison'])
        self.assertEqual(fields['frequency']['unit'],'seconds_per_ball')
        self.assertEqual(fields['direction']['value'],'正手定点')
        self.assertIsNone(fields['spin']['value'])
        self.assertIsNone(machine_fields({'ball_speed':90})['speed']['value'])

    def test_invalid_and_unknown_values_do_not_get_defaults(self):
        fields=machine_fields({'serve_interval':0,'serve_speed':True})
        self.assertIsNone(fields['frequency']['value'])
        self.assertIsNone(fields['speed']['value'])
        self.assertIsNone(timestamp('not-a-time'))

    def test_timezone_conversion_is_explicit(self):
        self.assertEqual(timestamp('2026-09-30T02:48:43Z'),timestamp('2026-09-30 10:48:43'))
