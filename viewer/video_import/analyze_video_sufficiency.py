"""Exhaustive video-subset support evidence; neither actual atlas nor fit accuracy.

The fixed cached training observations already exclude every 25th sampled frame.
Subset unions never modify observation thresholds or consult held-out colors.
"""
import argparse,csv,json,math
from pathlib import Path
import cv2,numpy as np
from multivideo_texture import uv_lookup
from run_records import sha256,write,now

def subset_support(support,uv_weights):
    count,faces=support.shape
    if count>20:raise ValueError('Exhaustive analysis is bounded to 20 clips')
    unions=np.zeros((1<<count,faces),bool)
    for mask in range(1,1<<count):
        bit=mask&-mask;unions[mask]=unions[mask^bit]|support[bit.bit_length()-1]
    return unions.sum(1),unions@uv_weights,unions

def smallest_subsets(rows,target,full_faces,full_uv):
    qualifying=[r for r in rows if r['supported_faces']>=target*full_faces and r['supported_uv_proxy_texels']>=target*full_uv]
    minimum=min(r['videos'] for r in qualifying)
    tied=[r for r in qualifying if r['videos']==minimum]
    best=sorted(tied,key=lambda r:(min(r['fraction_of_9clip_face_support'],r['fraction_of_9clip_uv_proxy_support']),r['supported_uv_proxy_texels']),reverse=True)
    return {'target_fraction_of_current_all_clip_support':target,'minimum_k':minimum,'qualifying_minimum_k_subsets':len(tied),'recommended':best[0],'all_minimum_k_subsets':best}

def analyze(observations,layout,source_index,output,size=2048):
    output.mkdir(parents=True,exist_ok=True)
    index=json.loads(source_index.read_text());clips=index['clips'];names=[Path(c['video']).stem for c in clips]
    baseline=json.loads((observations/'texture_report.json').read_text())
    if [r['video'] for r in baseline['validation']]!=names:raise ValueError('Observation ordering does not match source index')
    if sha256(layout)!=baseline['layout_sha256']:raise ValueError('UV layout hash changed')
    with np.load(layout,allow_pickle=False) as rig:
        face_map,_=uv_lookup(rig['uv'].astype(np.float32),rig['uv_faces'].astype(np.int32),size)
        faces=len(rig['faces'])
    weights=np.bincount(face_map[face_map>=0],minlength=faces).astype(np.int64);np.save(output/'uv_face_raster_weights.npy',weights,allow_pickle=False)
    support=[];inputs=[];heldout=[];tiles=[];sparse=[]
    for clip,name in zip(clips,names):
        cache=observations/(name+'_observations.npz');identity=json.loads(cache.with_suffix('.json').read_text());folder=Path(clip['folder']);video=folder/'source.mp4'
        if identity['step']!=5 or identity['mirror_source']!='independent':raise ValueError('Expected original fixed independent observations')
        if identity['video_sha256']!=clip['normalized_sha256'] or identity['video_sha256']!=sha256(video) or identity['archive_sha256']!=clip['archive_sha256']:raise ValueError('Source identity changed')
        with np.load(cache,allow_pickle=False) as d:
            available=np.any(d['all_score']>0,axis=0);support.append(available)
            if not np.array_equal(available,d['score']>0):raise ValueError('Cached best and union disagree')
            if np.any(d['all_frame']%25==0):raise ValueError('Heldout training leakage')
            htuple=np.c_[d['heldout_frame'],d['heldout_view'],d['heldout_face']].astype(np.int32)
            np.save(output/(name+'_fixed_heldout_tuples.npy'),htuple,allow_pickle=False)
            heldout.append({'video':name,'count':len(htuple),'tuple_sha256':sha256(output/(name+'_fixed_heldout_tuples.npy'))})
        inputs.append({'video':name,'observations_sha256':sha256(cache),'observation_identity':identity,'source_record':clip})
        cap=cv2.VideoCapture(str(video));row=[]
        for frame in [25,100,175]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
            if not ok:raise ValueError('Cannot decode sparse video evidence')
            thumb=cv2.resize(image,(480,270),interpolation=cv2.INTER_AREA);canvas=np.zeros((294,480,3),np.uint8);canvas[24:]=thumb
            cv2.putText(canvas,f'{name} / frame {frame} / {frame/25:.1f}s',(8,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,235,235),1,cv2.LINE_AA);row.append(canvas)
            sparse.append({'video':name,'normalized_zero_based_frame':frame,'time_seconds':frame/25,'normalized_video_sha256':identity['video_sha256']})
        cap.release();tiles.append(np.concatenate(row,axis=1))
    cv2.imwrite(str(output/'fixed_scene_contact_sheet.jpg'),np.concatenate(tiles,axis=0))
    support=np.array(support);face_counts,uv_counts,unions=subset_support(support,weights);full_faces=int(face_counts[-1]);full_uv=int(uv_counts[-1]);surface_uv=int(weights.sum());rows=[]
    for mask in range(1,len(face_counts)):
        selected=[names[i] for i in range(len(names)) if mask&(1<<i)]
        rows.append({'bitmask':mask,'videos':len(selected),'clips':selected,'supported_faces':int(face_counts[mask]),'face_support_fraction_of_all_mesh_faces':float(face_counts[mask]/faces),'supported_uv_proxy_texels':int(uv_counts[mask]),'uv_support_proxy_fraction_of_uv_surface':float(uv_counts[mask]/surface_uv),'fraction_of_9clip_face_support':float(face_counts[mask]/full_faces),'fraction_of_9clip_uv_proxy_support':float(uv_counts[mask]/full_uv)})
    by_k=[]
    for k in range(1,len(names)+1):
        selected=[r for r in rows if r['videos']==k];f=np.array([r['face_support_fraction_of_all_mesh_faces'] for r in selected]);u=np.array([r['uv_support_proxy_fraction_of_uv_surface'] for r in selected]);fr=np.array([r['fraction_of_9clip_face_support'] for r in selected]);ur=np.array([r['fraction_of_9clip_uv_proxy_support'] for r in selected])
        by_k.append({'k':k,'subsets':len(selected),'face_fraction':{'min':float(f.min()),'median':float(np.median(f)),'max':float(f.max())},'uv_support_proxy_fraction':{'min':float(u.min()),'median':float(np.median(u)),'max':float(u.max())},'relative_to_9clip_face_support':{'min':float(fr.min()),'median':float(np.median(fr)),'max':float(fr.max())},'relative_to_9clip_uv_proxy_support':{'min':float(ur.min()),'median':float(np.median(ur)),'max':float(ur.max())}})
    marginal=[];full_mask=(1<<len(names))-1
    for i,name in enumerate(names):
        bit=1<<i;shapley_face=shapley_uv=0.
        for mask in range(full_mask+1):
            if mask&bit:continue
            k=mask.bit_count();factor=math.factorial(k)*math.factorial(len(names)-k-1)/math.factorial(len(names))
            shapley_face+=factor*(float(face_counts[mask|bit])-float(face_counts[mask]));shapley_uv+=factor*(float(uv_counts[mask|bit])-float(uv_counts[mask]))
        marginal.append({'video':name,'standalone_faces':int(face_counts[bit]),'standalone_uv_proxy_texels':int(uv_counts[bit]),'leave_one_out_unique_faces':int(full_faces-face_counts[full_mask^bit]),'leave_one_out_unique_uv_proxy_texels':int(full_uv-uv_counts[full_mask^bit]),'leave_one_out_unique_uv_fraction_of_total_support':float((full_uv-uv_counts[full_mask^bit])/full_uv),'shapley_faces_average_over_all_orderings':shapley_face,'shapley_uv_proxy_texels_average_over_all_orderings':shapley_uv})
    greedy=[];mask=0
    while mask!=full_mask:
        options=[i for i in range(len(names)) if not mask&(1<<i)];take=max(options,key=lambda i:uv_counts[mask|(1<<i)]-uv_counts[mask]);new=mask|(1<<take)
        greedy.append({'k':new.bit_count(),'added':names[take],'uv_proxy_increment':int(uv_counts[new]-uv_counts[mask]),'face_increment':int(face_counts[new]-face_counts[mask]),'fraction_of_9clip_uv_proxy_support':float(uv_counts[new]/full_uv),'fraction_of_9clip_face_support':float(face_counts[new]/full_faces)});mask=new
    redundancy=np.bincount(support.sum(0),minlength=len(names)+1)
    report={'finished_at':now(),'status':'coverage_evidence_not_complete_fitting','code_sha256':sha256(Path(__file__)),'layout_sha256':sha256(layout),'source_index_sha256':sha256(source_index),'observation_report_sha256':sha256(observations/'texture_report.json'),'uv_face_weights_sha256':sha256(output/'uv_face_raster_weights.npy'),'inputs':inputs,'fixed_holdout':heldout,'heldout_rule':baseline['heldout_rule'],'training_rule':'Original cached all_score>0 only, no new masking/quality threshold, no heldout colors used','subsets_analyzed':len(rows),'mesh_faces':faces,'surface_uv_raster_texels':surface_uv,'uv_raster_size':size,'current_all_clip_support':rows[-1],'by_video_count':by_k,'minimum_subsets_95pct_current_support':smallest_subsets(rows,.95,full_faces,full_uv),'minimum_subsets_99pct_current_support':smallest_subsets(rows,.99,full_faces,full_uv),'video_marginal_contributions':marginal,'greedy_uv_support_order':greedy,'repetition':{'faces_supported_by_n_clips':redundancy.tolist(),'supported_faces_seen_in_at_least_5_clips_fraction':float((support.sum(0)>=5).sum()/full_faces),'sum_individual_face_support_divided_by_union':float(support.sum()/full_faces),'sum_individual_uv_proxy_support_divided_by_union':float((support@weights).sum()/full_uv)},'sparse_scene_evidence':{'contact_sheet_sha256':sha256(output/'fixed_scene_contact_sheet.jpg'),'samples':sparse,'stroke_classification':'not verified; no automatic forehand/backhand labels','visual_review_pending':True},'proxy_limits':'A face-center training visibility union weighted by MHR UV raster area is an optimistic support proxy. It is not the actual sampled texel coverage, calibrated fit accuracy, or body completeness. Center-visible faces may have occluded/invalid texels.'}
    actual=[]
    for label,path in [('first_clip_baseline',observations.parent/'baseline_independent/texture_report.json'),('old_nine_clip_atlas',observations/'texture_report.json'),('consistency_candidate',Path('output/multiagent_gpu/texture_quality/consistency_v1/texture_report.json'))]:
        if path.exists():
            r=json.loads(path.read_text());actual.append({'label':label,'report_path':str(path.resolve()),'report_sha256':sha256(path),'atlas_observed_texel_fraction':r['atlas_observed_texel_fraction'],'source':'previous actual atlas, not rerendered for this support analysis'})
    report['existing_actual_atlas_measurements']=actual
    write(output/'subset_results.json',rows);write(output/'sufficiency_report.json',report)
    with (output/'subset_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for r in rows:writer.writerow(dict(r,clips='|'.join(r['clips'])))
    with (output/'by_video_count.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['k','subsets','face_min','face_median','face_max','UV_support_proxy_min','UV_support_proxy_median','UV_support_proxy_max'])
        for r in by_k:writer.writerow([r['k'],r['subsets'],*[r['face_fraction'][v] for v in ['min','median','max']],*[r['uv_support_proxy_fraction'][v] for v in ['min','median','max']]])
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observations',type=Path,default=Path('output/multivideo_texture/fusion_independent'));p.add_argument('--layout',type=Path,default=Path('output/multivideo_texture/mhr_uv.npz'));p.add_argument('--source-index',type=Path,default=Path('output/multivideo_texture/source_index.json'));p.add_argument('--output',type=Path,default=Path('output/video_sufficiency'));a=p.parse_args();r=analyze(a.observations,a.layout,a.source_index,a.output);print(json.dumps({'minimum95':r['minimum_subsets_95pct_current_support']['recommended'],'minimum99':r['minimum_subsets_99pct_current_support']['recommended'],'by_k':r['by_video_count'],'greedy':r['greedy_uv_support_order']},indent=2))
