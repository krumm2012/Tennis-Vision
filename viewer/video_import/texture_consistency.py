"""Training-only source selection and fixed-denominator texture comparison.

A consensus color is a ranking reference, never a rendered pixel. Each observed
face retains every valid observation with positive score; unknown faces stay unknown.
"""
import argparse,json,warnings,shutil
from pathlib import Path
import numpy as np
from multivideo_texture import coherent_labels,face_neighbors,fuse
from run_records import sha256,write,now

CONSISTENCY_SCALE=24.0
SMOOTHNESS=.6

def consistency_scores(scores,colors,clip_ids):
    """Equal influence per clip prevents a long/repeated pose dominating consensus."""
    medians=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for clip in np.unique(clip_ids):
            selected=clip_ids==clip
            values=np.where((scores[selected]>0)[...,None],colors[selected],np.nan)
            medians.append(np.nanmedian(values,axis=0))
        consensus=np.nanmedian(np.stack(medians),axis=0)
    residual=np.abs(colors-consensus[None]).mean(2)
    # No threshold discards support; a very discrepant color remains an option.
    weight=np.maximum(np.exp(-np.nan_to_num(residual,nan=0)/CONSISTENCY_SCALE),.05)
    return (scores*weight).astype(np.float32),consensus

def summary(error):
    return {'count':len(error),'mean':float(np.mean(error)) if len(error) else None,'median':float(np.median(error)) if len(error) else None,'p95':float(np.percentile(error,95)) if len(error) else None}

def compare(observations,layout,baseline_report,output):
    source=json.loads(baseline_report.read_text());names=[v['video'] for v in source['validation']]
    for name,digest in source['artifact_sha256'].items():
        if sha256(observations/name)!=digest:raise ValueError('Baseline artifact hash mismatch: '+name)
    if sha256(layout)!=source['layout_sha256']:raise ValueError('Baseline UV layout hash mismatch')
    entries=[np.load(observations/(name+'_observations.npz'),allow_pickle=False) for name in names]
    identities=[json.loads((observations/(name+'_observations.json')).read_text()) for name in names]
    scores=np.concatenate([d['all_score'] for d in entries]);clips=np.concatenate([np.full(len(d['all_score']),i) for i,d in enumerate(entries)])
    gains=[np.array(v['gain_bgr']) for v in source['validation']]
    colors=np.concatenate([d['all_color'].astype(np.float32)*gains[k] for k,d in enumerate(entries)])
    with np.load(layout,allow_pickle=False) as rig:faces=rig['faces']
    ids=np.arange(len(faces));baseline=coherent_labels(scores,faces)
    adjusted,consensus=consistency_scores(scores,colors,clips);candidate=coherent_labels(adjusted,faces,smoothness=SMOOTHNESS)
    base_support=scores[baseline,ids]>0;candidate_support=scores[candidate,ids]>0
    base_colors=colors[baseline,ids];candidate_colors=colors[candidate,ids];rows=[];base_all=[];candidate_all=[]
    for i,d in enumerate(entries):
        # These exact held-out tuples are never used in consensus/ranking.
        hf=d['heldout_face'];fixed=base_support[hf];common=fixed&candidate_support[hf]
        observed=d['heldout_color'].astype(np.float32)*gains[i]
        base_error=np.abs(base_colors[hf[fixed]]-observed[fixed]).mean(1)
        candidate_error=np.full(fixed.sum(),255.,np.float64)
        candidate_error[candidate_support[hf[fixed]]]=np.abs(candidate_colors[hf[common]]-observed[common]).mean(1)
        base_common=np.abs(base_colors[hf[common]]-observed[common]).mean(1)
        candidate_common=np.abs(candidate_colors[hf[common]]-observed[common]).mean(1)
        tuples=np.c_[np.full(len(hf),i),d['heldout_frame'],d['heldout_view'],hf].astype(np.int32)
        np.save(output/(names[i]+'_heldout_tuples.npy'),tuples,allow_pickle=False)
        rows.append({'video':names[i],'all_observed_heldout_tuples':len(hf),'fixed_baseline_supported_count':int(fixed.sum()),'common_valid_count':int(common.sum()),'missing_candidate_penalty_0_255':255,'baseline_fixed':summary(base_error),'candidate_fixed':summary(candidate_error),'baseline_common':summary(base_common),'candidate_common':summary(candidate_common),'paired_mean_delta_candidate_minus_baseline':float((candidate_error-base_error).mean()),'heldout_tuple_sha256':sha256(output/(names[i]+'_heldout_tuples.npy'))})
        base_all.append(base_error);candidate_all.append(candidate_error)
    neighbors=face_neighbors(faces);edges=np.c_[np.repeat(ids,3),neighbors.ravel()];edges=edges[(edges[:,1]>=0)&(edges[:,0]<edges[:,1])];edges=edges[base_support[edges].all(1)&candidate_support[edges].all(1)]
    def edge_metrics(labels,color):
        return {'fixed_common_edges':len(edges),'view_transition_fraction':float((labels[edges[:,0]]!=labels[edges[:,1]]).mean()),'neighbor_color_mae_0_255':summary(abs(color[edges[:,0]]-color[edges[:,1]]).mean(1))}
    base_all=np.concatenate(base_all);candidate_all=np.concatenate(candidate_all)
    np.savez_compressed(output/'selection_comparison.npz',baseline_label=baseline,candidate_label=candidate,baseline_support=base_support,candidate_support=candidate_support,training_consensus_bgr=consensus)
    report={'finished_at':now(),'status':'candidate_comparison_not_geometry_validation','baseline_texture_report_sha256':sha256(baseline_report),'baseline_artifact_sha256':source['artifact_sha256'],'layout_sha256':sha256(layout),'observation_code_sha256':[x['code_sha256'] for x in identities],'observation_sha256':{n:sha256(observations/(n+'_observations.npz')) for n in names},'selection_code_sha256':sha256(Path(__file__)),'fusion_code_sha256':sha256(Path(__file__).with_name('multivideo_texture.py')),'inputs':identities,'parameters':{'color_consistency_scale_bgr':CONSISTENCY_SCALE,'score_floor_multiplier':.05,'edge_potts_weight':SMOOTHNESS,'baseline_edge_potts_weight':.25,'iterations':5},'training_only':True,'selection_colors_rendered':False,'face_support':{'baseline':int(base_support.sum()),'candidate':int(candidate_support.sum()),'common':int((base_support&candidate_support).sum()),'lost':int((base_support&~candidate_support).sum())},'fixed_all_clips_baseline':summary(base_all),'fixed_all_clips_candidate':summary(candidate_all),'paired_mean_delta_candidate_minus_baseline':float((candidate_all-base_all).mean()),'baseline_edges':edge_metrics(baseline,base_colors),'candidate_edges':edge_metrics(candidate,candidate_colors),'per_clip':rows,'metric_limits':'Fixed face-center appearance residual; neighboring colors can contain real clothing edges. Neither metric is novel-view or body geometry accuracy. UV coverage is reported separately.'}
    write(output/'comparison_report.json',report)
    for d in entries:d.close()
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--layout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=2048);p.add_argument('--compare-only',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    baseline=args.output/'baseline';baseline.mkdir(exist_ok=True)
    for name in ['body_texture_rgba.png','texture_sources.npz','appearance_mesh.npz','texture_report.json']:shutil.copy2(args.observations/name,baseline/name)
    comparison=compare(args.observations,args.layout,args.observations/'texture_report.json',args.output)
    if not args.compare_only:
        candidate=fuse(args.batch,args.layout,args.output,args.size,5,'independent','consistency',args.observations)
        import cv2
        baseline_image=cv2.imread(str(args.observations/'body_texture_rgba.png'),cv2.IMREAD_UNCHANGED);candidate_image=cv2.imread(str(args.output/'body_texture_rgba.png'),cv2.IMREAD_UNCHANGED)
        if baseline_image.shape!=candidate_image.shape:raise ValueError('Fixed-denominator atlas requires identical size')
        baseline_known=baseline_image[:,:,3]>0;candidate_known=candidate_image[:,:,3]>0
        sources=np.load(args.observations/'texture_sources.npz',allow_pickle=False);surface=sources['face']>=0
        comparison['atlas_fixed_denominator']={'surface_texels':int(surface.sum()),'baseline_observed':int((surface&baseline_known).sum()),'candidate_observed':int((surface&candidate_known).sum()),'common_observed':int((surface&baseline_known&candidate_known).sum()),'lost_observed':int((surface&baseline_known&~candidate_known).sum()),'newly_observed':int((surface&~baseline_known&candidate_known).sum()),'baseline_coverage':float(baseline_known[surface].mean()),'candidate_coverage':float(candidate_known[surface].mean())}
        comparison['candidate_artifact_sha256']=candidate['artifact_sha256'];sources.close();write(args.output/'comparison_report.json',comparison)
    print(json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__':main()
