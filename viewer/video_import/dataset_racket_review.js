// Sparse, video-scoped observations. Occluded landmarks stay absent.
(()=>{
'use strict';
const names=[['handle_end','握柄末端'],['throat','拍喉'],['tip','拍头'],['rim_side','拍框 A'],['rim_opposite','拍框 B']];
const ident=location.pathname.match(/\/datasets\/([0-9a-f]{32})\//)?.[1];if(!ident)return;
let saved=[],draft,editingFrame=0,view='points',part='handle_end';
const controls=document.createElement('div');controls.className='tool-row';controls.innerHTML='<button id="racketReviewOpen" title="只标清晰关键帧，遮挡点留空">◎ 球拍校准</button><label><input id="racketEvidence" type="checkbox">方向观测</label><label>版本 <select id="racketVersion" disabled><option value="current">当前默认</option><option value="directional">方向候选 · 待复核</option></select></label><span id="racketReviewStatus" role="status"></span>';
document.querySelector('#datasetTools').append(controls);
const dialog=document.createElement('dialog');dialog.innerHTML=`<h2>◎ 球拍关键帧</h2><div class="tool-row"><label>观测 <select id="reviewView"><option value="points">真人</option><option value="mirror_points">镜中</option></select></label><label>点 <select id="reviewPart">${names.map(([k,n])=>`<option value="${k}">${n}</option>`).join('')}</select></label><button id="reviewClear">清除点</button><button id="reviewDelete">删除本帧</button></div><canvas id="reviewCanvas" tabindex="0" aria-label="球拍关键点标注"></canvas><div class="tool-row"><label><input id="reviewFace" type="checkbox">A 侧固定物理标记已核对</label><input id="reviewFeature" maxlength="160" placeholder="例如：A 侧蓝色贴纸" aria-label="固定物理标记说明"></div><small>只标可见点。镜中 A/B 须对应相同物理侧。无遮挡且 A 侧可识别时，才确认有向法线；未知正反面保留待校准。</small><p id="reviewStatus" role="status"></p><div class="tool-row"><button id="reviewSave">✓ 保存并拟合</button><button id="reviewClose">关闭</button></div>`;
document.body.append(dialog);const image=$('reviewCanvas'),context=image.getContext('2d'),still=document.createElement('canvas');
const style=document.createElement('style');style.textContent='#reviewCanvas{height:auto;max-height:60vh;object-fit:contain;cursor:crosshair}#racketReviewStatus{font-size:13px;color:#a8bfcc}';document.head.append(style);
const payload=()=>({video_sha256:renderer.meta.video_sha256,image_size:renderer.meta.image_size,fps:renderer.meta.fps,frames:saved});
function repaint(){context.drawImage(still,0,0);for(const key of ['points','mirror_points']){const pts=draft[key];context.lineWidth=3;context.strokeStyle=key==='points'?'#ffcf6b':'#63e5db';if(pts.handle_end&&pts.tip){context.beginPath();context.moveTo(...pts.handle_end);context.lineTo(...pts.tip);context.stroke()}for(const [name,label] of names){if(!pts[name])continue;context.beginPath();context.arc(...pts[name],name===part&&view===key?9:6,0,Math.PI*2);context.fillStyle=context.strokeStyle;context.fill();context.font='bold 18px system-ui';context.fillText(label,pts[name][0]+12,pts[name][1]-10)}}$('reviewStatus').textContent=`${editingFrame+1}/${renderer.meta.frames} 帧 · 真人 ${Object.keys(draft.points).length}/5 · 镜中 ${Object.keys(draft.mirror_points).length}/5`;}
$('racketReviewOpen').onclick=()=>{if(!renderer?.ready||video.seeking||video.readyState<2)return;pause();editingFrame=frame;draft=structuredClone(saved.find(r=>r.frame===frame)||{frame,points:{},mirror_points:{},face_correspondence_confirmed:false,side_feature:''});view='points';part='handle_end';$('reviewView').value=view;$('reviewPart').value=part;$('reviewFace').checked=!!draft.face_correspondence_confirmed;$('reviewFeature').value=draft.side_feature||'';[image.width,image.height]=renderer.meta.image_size;still.width=image.width;still.height=image.height;still.getContext('2d').drawImage(video,0,0,still.width,still.height);dialog.showModal();repaint();image.focus()};
$('reviewView').onchange=e=>{view=e.target.value;repaint()};$('reviewPart').onchange=e=>{part=e.target.value;repaint()};
image.onclick=e=>{const r=image.getBoundingClientRect(),s=Math.min(r.width/image.width,r.height/image.height),x=(e.clientX-r.left-(r.width-image.width*s)/2)/s,y=(e.clientY-r.top-(r.height-image.height*s)/2)/s;if(x<0||y<0||x>image.width||y>image.height)return;draft[view][part]=[x,y];repaint();image.focus()};
image.onkeydown=e=>{const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(!d||!draft[view][part])return;e.preventDefault();draft[view][part]=draft[view][part].map((v,i)=>Math.max(0,Math.min(i?image.height:image.width,v+d[i]*(e.shiftKey?10:1))));repaint()};
$('reviewClear').onclick=()=>{delete draft[view][part];repaint()};$('reviewDelete').onclick=()=>{draft.points={};draft.mirror_points={};$('reviewFace').checked=false;repaint()};$('reviewClose').onclick=()=>dialog.close();
$('reviewSave').onclick=async()=>{const button=$('reviewSave');button.disabled=true;try{draft.face_correspondence_confirmed=$('reviewFace').checked;draft.side_feature=$('reviewFeature').value.trim();const next=saved.filter(r=>r.frame!==editingFrame);if(Object.keys(draft.points).length||Object.keys(draft.mirror_points).length)next.push(draft);$('reviewStatus').textContent='正在拟合关键点…';const response=await fetch(`/api/videos/${ident}/racket-landmarks`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload(),frames:next})});$('reviewStatus').textContent='正在拟合关键点…';const result=await response.json();if(!response.ok)throw Error(result.error||'拟合失败');saved=next;dialog.close();await refreshQuality();await loadVersion($('racketVersion').value);}catch(e){$('reviewStatus').textContent=e.message;if(!dialog.open)$('racketReviewStatus').textContent=e.message}finally{button.disabled=false}};
$('racketEvidence').onchange=e=>{if(window.datasetRacket)window.datasetRacket.showEvidence=e.target.checked;draw()};
async function refreshQuality(){
 let quality=null;try{const response=await fetch('racket_quality_gate.json',{cache:'no-store'});if(response.ok)quality=await response.json()}catch{}
 const select=$('racketVersion');select.disabled=!quality?.candidate_sha256;
 select.options[1].textContent=quality?.status==='accepted'?'方向候选 · 已通过一致性检查':'方向候选 · 待复核';
 $('racketReviewStatus').textContent=quality?.status==='needs_review'?'候选未通过画面一致性检查 · 默认保留':quality?.status==='accepted'?'方向拟合通过一致性检查':quality?'方向候选缺少对照 · 待复核':saved.length?`${saved.length} 个校准关键帧`:'关键帧校准 · 法线正负待确认';
}
async function loadVersion(version){
 const select=$('racketVersion'),old=window.datasetRacket,next=new DatasetRacketLayer();pause();select.disabled=true;
 try{
  await next.load(renderer,version==='directional'?'racket_poses_directional.json':'racket_poses.json');if(!next.state)throw Error('球拍版本不可用');
  next.enabled=$('racketEnabled').checked;next.showEvidence=$('racketEvidence').checked;
  if(old?.state){renderer.scene.remove(old.state.mesh);old.dispose();renderer.layers=renderer.layers.filter(l=>l!==old)}window.datasetRacket=next;
  const s=next.summary;$('racketStatus').textContent=`${version==='directional'?'方向候选':'当前默认'} · 观测 ${s.observed} · 估计 ${s.interpolated} · 隐藏 ${s.hidden} 帧${s.stereo_shaft_frames?` · 双视角拍柄 ${s.stereo_shaft_frames}`:''}${s.direction_review_frames?` · 方向待复核 ${s.direction_review_frames}`:''} · 握柄棱位待确认`;draw();
 }catch(e){if(next.state)renderer.scene.remove(next.state.mesh);next.dispose();renderer.layers=renderer.layers.filter(l=>l!==next);select.value=old?.poseFile==='racket_poses_directional.json'?'directional':'current';throw e}
 finally{select.disabled=false}
}
$('racketVersion').onchange=async e=>{try{await loadVersion(e.target.value)}catch(error){$('racketReviewStatus').textContent=error.message}};
window.initDatasetRacketReview=async()=>{try{const response=await fetch('racket_landmarks.json',{cache:'no-store'});if(response.ok){const data=await response.json();if(data.video_sha256===renderer.meta.video_sha256)saved=data.frames}}catch{}await refreshQuality()};
})();
