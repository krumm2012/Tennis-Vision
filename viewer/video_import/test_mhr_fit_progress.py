import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mhr_fit_progress import collect, record_fit, record_native_audit
from run_records import sha256, write
import run_fullbody_refit


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.out=self.root/'fit'; self.out.mkdir()
        self.ledger=self.root/'ledger'

    def fixture(self, gate=False):
        write(self.out/'remote_job.json',{'job_id':'8'*32,'parent_job':'c'*32})
        write(self.out/'readiness.json',{'ready':True,'video_sha256':'v'*64,'archive_sha256':'a'*64,
              'train_frames':[1,2],'heldout_frames':[3],'validation_source':'automatic_contours_unverified',
              'assumed_dimensions_permitted':True,'size_measured':False,'camera_calibrated':False})
        candidate=self.out/'mhr_refit_candidate.npz'; candidate.write_bytes(b'candidate')
        self.report={'full_body_joint_fit_completed':True,'steps':120,'numerical_regression_gate_passed':gate,
                     'native_replay_mesh_max_m':7e-7,'native_replay_joint_max_m':7e-7,
                     'body_displacement_p95_m':.01,'heldout_landmark_before_canonical_px':15.2,
                     'heldout_landmark_after_canonical_px':15.3,'camera_roots_preserved':True,'shape_and_scale_preserved':True,
                     'dimensions_measured':False,'physical_grip_bevel_verified':False,
                     'contact_method':'joint/cylinder proxy; not mesh surface contact','validation_source':'automatic_contours_unverified',
                     'video_sha256':'v'*64,'source_archive_sha256':'a'*64,'code_sha256':'f'*64,'candidate_sha256':sha256(candidate)}
        write(self.out/'refit_report.json',self.report)

    def events(self):
        return [json.loads(p.read_text()) for p in (self.ledger/'events').glob('*.json')]

    def test_computed_fit_with_failed_gate_is_never_accepted(self):
        self.fixture()
        c=record_fit(self.out,self.ledger)
        self.assertEqual(c['execution']['state'],'completed')
        self.assertFalse(c['acceptance']['accepted_joint_fit'])
        self.assertEqual(c['acceptance']['state'],'rejected')
        self.assertEqual(c['stages']['joint_optimisation']['steps'],120)
        self.assertEqual(c['stages']['heldout_landmark_regression']['state'],'failed')
        for name in ['hand_mesh_contact','independent_observations','shared_identity_optimization','camera_and_scale_calibration']:
            self.assertEqual(c['stages'][name]['state'],'not_completed')
        self.assertEqual(c['dimension_status'],'assumed')
        self.assertEqual(c['readiness']['heldout_frames'],[3])

    def test_failure_history_survives_later_report_for_same_run(self):
        write(self.out/'remote_job.json',{'job_id':'2'*32})
        write(self.out/'run_manifest.json',{'status':'failed','remote_job_id':'2'*32})
        (self.out/'worker.log').write_text('ValueError: 重推理相机与现有标定不一致')
        first=record_fit(self.out,self.ledger)
        self.assertEqual(first['execution']['state'],'failed')
        self.assertEqual(first['stages']['camera_consistency']['state'],'failed')
        self.fixture(); write(self.out/'remote_job.json',{'job_id':'2'*32})
        second=record_fit(self.out,self.ledger)
        self.assertNotEqual(first['event_id'],second['event_id'])
        self.assertEqual(len(self.events()),2)
        self.assertEqual(json.loads(Path(first['ledger_event']).read_text())['content']['execution']['state'],'failed')

    def test_provenance_is_idempotent_and_report_mutation_adds_event(self):
        self.fixture()
        first=record_fit(self.out,self.ledger); again=record_fit(self.out,self.ledger)
        self.assertEqual(first['event_id'],again['event_id']); self.assertEqual(len(self.events()),1)
        first_hash=first['provenance']['source_files']['refit_report.json']['sha256']
        self.report['steps']=121;write(self.out/'refit_report.json',self.report)
        changed=record_fit(self.out,self.ledger)
        self.assertNotEqual(changed['event_id'],first['event_id'])
        self.assertNotEqual(changed['provenance']['source_files']['refit_report.json']['sha256'],first_hash)
        self.assertEqual(len(self.events()),2)

    def test_native_audit_does_not_count_as_joint_fit(self):
        path=self.root/'audit.json';write(path,{'audit_id':'audit','status':'complete','clips':[]})
        c=record_native_audit(path,self.ledger)
        self.assertEqual(c['stages']['native_replay_audit']['state'],'unknown')
        ledger=json.loads((self.ledger/'ledger.json').read_text())
        self.assertEqual(ledger['accepted_joint_fit_runs'],0)
        self.assertEqual(ledger['runs'][0]['event_kind'],'native_replay_audit')
        self.assertEqual(ledger['runs'][0]['latest_acceptance'],'not_applicable')

    def test_completed_audit_requires_per_clip_verification_evidence(self):
        path=self.root/'audit.json'
        clip={'status':'verified_replay','source_hash_matches_request':True,
              'topology_matches_provisioned_model':True,'embedded_video_hash_matches':True,
              'frame_count_matches_manifest':True,'real':{'replay_gate_passed':True},'mirror':{'replay_gate_passed':False}}
        report={'audit_id':'audit','status':'complete','clips':[clip]}
        write(path,report);failed=record_native_audit(path,self.ledger)
        self.assertEqual(failed['execution']['state'],'completed')
        self.assertEqual(failed['stages']['native_replay_audit']['state'],'failed')
        clip['mirror']['replay_gate_passed']=True;del clip['topology_matches_provisioned_model']
        write(path,report);unknown=record_native_audit(path,self.ledger)
        self.assertEqual(unknown['stages']['native_replay_audit']['state'],'unknown')
        clip['topology_matches_provisioned_model']=True
        write(path,report);passed=record_native_audit(path,self.ledger)
        self.assertEqual(passed['stages']['native_replay_audit']['state'],'passed')
        self.assertEqual(len(self.events()),3)
        self.assertFalse(passed['acceptance']['accepted_joint_fit'])

    def test_needs_input_runner_path_is_recorded_without_dispatch(self):
        with patch.object(run_fullbody_refit,'assess',return_value={'ready':False,'blocked_by':['reviewed_frames_missing']}),patch.object(run_fullbody_refit,'ssh') as ssh:
            result=run_fullbody_refit.run(self.root,self.root,self.out,{},progress_ledger=self.ledger)
        ssh.assert_not_called();self.assertFalse(result['ready'])
        c=json.loads((self.out/'completion.json').read_text())
        self.assertEqual(c['execution']['state'],'needs_input')
        self.assertFalse(c['acceptance']['accepted_joint_fit'])
        self.assertTrue(any(e['content']['execution']['state']=='pending' for e in self.events()))

    def test_no_dispatch_recovery_uses_current_attempt_not_old_job(self):
        self.fixture()
        with patch.object(run_fullbody_refit,'assess',return_value={'ready':False,'blocked_by':['need_review']}):
            run_fullbody_refit.run(self.root,self.root,self.out,{},progress_ledger=self.ledger)
        first=json.loads((self.out/'completion.json').read_text())
        recovered=record_fit(self.out,self.ledger)
        self.assertEqual(first['run_id'],recovered['run_id'])
        self.assertNotEqual(recovered['run_id'],'8'*32)
        self.assertEqual(first['event_id'],recovered['event_id'])
        self.assertFalse(recovered['execution']['computed_fit'])
        self.assertNotIn('refit_report.json',recovered['provenance']['source_files'])

    def test_dispatched_run_and_cli_recovery_share_identity_and_event(self):
        native=self.root/'native';native.mkdir();archive=native/'reconstruction.npz';archive.write_bytes(b'native')
        write(native/'remote_job.json',{'job_id':'c'*32})
        write(native/'remote_manifest.json',{'status':'ready','source_sha256':'v'*64,'artifact_sha256':{'reconstruction.npz':sha256(archive)}})
        ready={'ready':True,'video_sha256':'v'*64,'archive_sha256':sha256(archive)}
        candidate=b'candidate';candidate_file=self.root/'candidate';candidate_file.write_bytes(candidate)
        report={'full_body_joint_fit_completed':True,'steps':120,'numerical_regression_gate_passed':False,
                'video_sha256':'v'*64,'source_archive_sha256':sha256(archive),'candidate_sha256':sha256(candidate_file)}
        def remote(c,args):
            return json.dumps({'status':'ready'}) if 'status' in args else ''
        def download(c,source,dest,upload=False):
            if upload:return
            if str(dest).endswith('remote_manifest.json'):
                job=json.loads((self.out/'remote_job.json').read_text())['job_id']
                write(dest,{'status':'ready','remote_job_id':job,'source_sha256':'v'*64,'refit':report,'artifact_sha256':{'mhr_refit_candidate.npz':sha256(candidate_file)}})
            elif str(dest).endswith('.npz.part'):Path(dest).write_bytes(candidate)
            elif str(dest).endswith('refit_report.json'):write(dest,report)
            elif str(dest).endswith('worker.log'):Path(dest).write_text('completed 120 steps')
        with patch.object(run_fullbody_refit,'assess',return_value=ready),patch.object(run_fullbody_refit,'ssh',side_effect=remote),patch.object(run_fullbody_refit,'copy',side_effect=download):
            run_fullbody_refit.run(self.root,native,self.out,{'root':'/private/gpu','python':'/private/python'},progress_ledger=self.ledger)
        first=json.loads((self.out/'completion.json').read_text())
        recovered=record_fit(self.out,self.ledger)
        self.assertEqual(first['run_id'],json.loads((self.out/'remote_job.json').read_text())['job_id'])
        self.assertEqual(first['run_id'],recovered['run_id'])
        self.assertEqual(first['event_id'],recovered['event_id'])
        self.assertEqual(len(json.loads((self.ledger/'ledger.json').read_text())['runs']),1)

    def test_failed_transport_after_dispatch_remains_remote_pending(self):
        def lost_connection(dataset,native,out,c,assumed,automatic,context,checkpoint):
            context.update(dispatch_attempted=True,dispatch_acknowledged=True,remote_job_id='r'*32)
            raise subprocess.CalledProcessError(255,['ssh'])
        with patch.object(run_fullbody_refit,'_execute',side_effect=lost_connection):
            with self.assertRaises(subprocess.CalledProcessError):
                run_fullbody_refit.run(self.root,self.root,self.out,{},progress_ledger=self.ledger)
        c=json.loads((self.out/'completion.json').read_text())
        self.assertEqual(c['execution']['state'],'transport_pending')
        self.assertEqual(c['execution']['error']['type'],'CalledProcessError')
        self.assertFalse(c['acceptance']['accepted_joint_fit'])

    def test_ledger_failure_does_not_mask_original_transport_exception(self):
        original=subprocess.CalledProcessError(255,['ssh'])
        with patch.object(run_fullbody_refit,'_execute',side_effect=original),patch.object(run_fullbody_refit,'record_fit',side_effect=OSError('disk unavailable')):
            with self.assertWarns(RuntimeWarning):
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    run_fullbody_refit.run(self.root,self.root,self.out,{},progress_ledger=self.ledger)
        self.assertIs(caught.exception,original)

    def test_failed_local_preparation_is_recorded_and_raised(self):
        with patch.object(run_fullbody_refit,'assess',side_effect=ValueError('invalid native source')):
            with self.assertRaises(ValueError):run_fullbody_refit.run(self.root,self.root,self.out,{},progress_ledger=self.ledger)
        c=json.loads((self.out/'completion.json').read_text())
        self.assertEqual(c['execution']['state'],'failed')
        self.assertFalse(c['execution']['computed_fit'])

    def test_remote_failed_return_keeps_downloaded_failure_evidence(self):
        native=self.root/'native';native.mkdir();archive=native/'reconstruction.npz';archive.write_bytes(b'native')
        write(native/'remote_job.json',{'job_id':'c'*32})
        write(native/'remote_manifest.json',{'status':'ready','source_sha256':'v'*64,'artifact_sha256':{'reconstruction.npz':sha256(archive)}})
        ready={'ready':True,'video_sha256':'v'*64,'archive_sha256':sha256(archive)}
        def remote(c,args):
            return json.dumps({'status':'failed'}) if 'status' in args else ''
        def download(c,source,dest,upload=False):
            if upload:return
            if str(dest).endswith('remote_manifest.json'):write(dest,{'status':'failed','error_type':'CalledProcessError'})
            if str(dest).endswith('worker.log'):Path(dest).write_text('ValueError: 重推理相机与现有标定不一致')
        with patch.object(run_fullbody_refit,'assess',return_value=ready),patch.object(run_fullbody_refit,'ssh',side_effect=remote),patch.object(run_fullbody_refit,'copy',side_effect=download):
            result=run_fullbody_refit.run(self.root,native,self.out,{'root':'/private/gpu','python':'/private/python'},progress_ledger=self.ledger)
        self.assertEqual(result['status'],'failed')
        c=json.loads((self.out/'completion.json').read_text())
        self.assertEqual(c['execution']['state'],'failed')
        self.assertEqual(c['stages']['camera_consistency']['state'],'failed')
        self.assertIn('worker.log',c['provenance']['source_files'])

    def test_old_output_artifacts_are_not_attributed_to_a_new_attempt(self):
        self.fixture()
        c=collect(self.out,context={'run_id':'new-attempt','current_files':[]},execution_state='pending')
        self.assertFalse(c['execution']['computed_fit']);self.assertIsNone(c['provenance']['candidate_sha256'])
        self.assertEqual(c['provenance']['source_files'],{})

    def test_candidate_hash_corruption_rejects_even_if_numerical_gate_passes(self):
        self.fixture(gate=True);(self.out/'mhr_refit_candidate.npz').write_bytes(b'corrupted')
        c=record_fit(self.out,self.ledger)
        self.assertEqual(c['stages']['candidate_integrity']['state'],'failed')
        self.assertEqual(c['acceptance']['state'],'rejected')

    def test_nonfinite_report_preserves_hash_and_does_not_pass(self):
        self.fixture();self.report['body_displacement_p95_m']=float('nan');write(self.out/'refit_report.json',self.report)
        c=record_fit(self.out,self.ledger)
        self.assertEqual(c['stages']['body_displacement_regression']['state'],'failed')
        self.assertEqual(c['stages']['body_displacement_regression']['p95_m'],{'nonfinite_value':'nan'})
        self.assertEqual(c['provenance']['source_files']['refit_report.json']['sha256'],sha256(self.out/'refit_report.json'))

    def test_current_manifest_does_not_adopt_older_report_and_candidate(self):
        self.fixture()
        run_id='9'*32
        write(self.out/'progress_attempt.json',{'run_id':run_id,'remote_job_id':run_id,'output':str(self.out.resolve())})
        write(self.out/'remote_job.json',{'job_id':run_id})
        write(self.out/'remote_manifest.json',{'remote_job_id':run_id,'status':'ready',
              'artifact_sha256':{'mhr_refit_candidate.npz':'b'*64}})
        c=collect(self.out)
        self.assertEqual(c['run_id'],run_id)
        self.assertFalse(c['execution']['computed_fit'])
        self.assertIsNone(c['provenance']['candidate_sha256'])
        self.assertNotIn('refit_report.json',c['provenance']['source_files'])


if __name__=='__main__':unittest.main()
