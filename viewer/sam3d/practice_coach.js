/* Ball-machine practice: explicit coach ratings with source-bound saved evidence. */
(function(root){
async function mount({container,video,seek}){
 const match=location.pathname.match(/^\/datasets\/([a-f0-9]{32})\/result\//);if(!match)return;
 const panel=document.createElement('section');panel.className='coach-panel practice-panel';container.append(panel);
 const style=document.createElement('style');style.textContent='.practice-panel{margin:16px 0;padding:18px;border:1px solid #405c79;border-radius:12px;background:#192638;color:#e6eef8}.practice-panel label{display:inline-block;margin:6px 10px 6px 0}.practice-panel form>div label{display:flex;align-items:center;gap:12px;max-width:850px}.practice-panel select{max-width:100%;min-width:140px;background:#253a50;color:#e6eef8;padding:8px;border:1px solid #587894;border-radius:7px}.practice-panel input,.practice-panel textarea{box-sizing:border-box;background:#101d2b;color:#e6eef8;border:1px solid #587894;border-radius:6px;padding:8px}.practice-panel button{font:inherit;background:#2d4560;color:#e6eef8;border:1px solid #587894;border-radius:7px;padding:8px;margin:4px;cursor:pointer}.practice-panel button:disabled{opacity:.5}.practice-panel p{line-height:1.6}.practice-panel small{color:#b6c9de}@media(max-width:700px){.practice-panel form>div label{display:block}.practice-panel select{width:100%}.practice-panel input[type=text]{max-width:100%}}';document.head.append(style);
 const el=(tag,text,parent=panel)=>{const n=document.createElement(tag);n.textContent=text;parent.append(n);return n};
 el('h2','发球机练习 · 教练评价');
 el('p','统一五项标准，每项 1–5 分。未评价留空；五项完成并由教练确认后显示总分。自动 3D 评分尚待校准，握拍拟合与自动测距另行验收。');
 const status=el('p','加载评分标准…');status.setAttribute('role','status');status.setAttribute('aria-live','polite');
 try{
  const endpoint=`/api/videos/${match[1]}/practice-review`;
  const [pr,sr]=await Promise.all([fetch('/api/practice-policy'),fetch(endpoint)]);
  if(!pr.ok||!sr.ok)throw Error('评分服务尚未就绪，请稍后刷新');
  const policy=await pr.json();let state=await sr.json(),selected=null;
  el('small',`${policy.version} · 草案，待独立教练样本校准 · 自动参考分与教练分分开比较`);
  const training=state.training_context;
  if(training){
   const details=el('details','');el('summary','发球机训练数据关联',details);
   const source=training.source||{},capture=training.capture_time||{};
   const message=training.status==='no_matching_records'?'暂无对应训练记录；发球机条件留空':'找到时间候选，需确认相机/设备与时间偏移后才能关联';
   el('p',`GW 用户 ${training.query_user_masked} · 返回 ${source.records_count} 条训练记录 · ${message}`);
   el('p',`视频时间候选：${capture.local_start||'未知'} 至 ${capture.local_end||'未知'}；原文件时间尚需与设备时钟核对。`,details);
   el('p',`该用户记录截至 ${source.last_record_at||'未知'}。设备得分与球速另行解释，不转换为动作评分。`,details);
   if((source.missing_detail_fields||[]).length)el('p','当前在线查询未返回完整发球明细和设置，不能自动填写速度、方向或旋转。',details);
   const a=el('a','查看匹配依据 JSON',details);a.href='machine_training_context.json';
  }
  const rows=el('div','');const form=el('form','');form.onsubmit=e=>e.preventDefault();
  const input=(label,type='text',parent=form)=>{const l=el('label',label+' ',parent),i=document.createElement('input');i.type=type;l.append(i);i.setAttribute('aria-label',label);return i};
  const start=input('开始帧','number'),end=input('结束帧','number');for(const i of [start,end]){i.min=0;i.max=state.frames-1;i.step=1;i.style.width='90px'}
  const button=(name,fn,parent=form)=>{const b=el('button',name,parent);b.type='button';b.onclick=fn;return b};
  const current=()=>Math.min(state.frames-1,Math.max(0,Math.round(video.currentTime*state.fps)));
  button('当前帧设为开始',()=>start.value=current());button('当前帧设为结束',()=>end.value=current());
  button('定位开始',()=>{if(start.value!=='')seek(Number(start.value)/state.fps)});
  const sl=el('label','动作 ',form),stroke=el('select','',sl);stroke.setAttribute('aria-label','动作类型');
  for(const [key,label] of [['Unknown','待确认'],['Forehand','正手'],['Backhand','单手反手'],['Two-Handed Backhand','双手反手'],['Volley','截击'],['Serve','发球']]){const o=el('option',label,stroke);o.value=key}
  el('br','',form);const goal=input('练习目标');goal.maxLength=200;const reviewer=input('教练姓名');reviewer.maxLength=80;
  const machine={};el('p','发球机条件（未知可留空；条件齐全后才适合跨片段比较）',form);
  for(const [key,label] of [['speed','来球速度（注明单位）'],['frequency','来球频率（注明单位）'],['direction','方向'],['spin','旋转']]){machine[key]=input(label);machine[key].maxLength=100}
  const ratingInputs={};
  for(const d of policy.dimensions){const row=el('div','',form),l=el('label',d.label+' ',row),select=el('select','',l);select.setAttribute('aria-label',d.label+'评分');const blank=el('option','未评价',select);blank.value='';for(let n=1;n<=5;n++){const o=el('option',`${n} · ${d.anchors[n-1]}`,select);o.value=n}ratingInputs[d.id]=select;}
  const ol=el('label','评价依据与下一步练习建议 ',form),observation=el('textarea','',ol);observation.setAttribute('aria-label','评价依据');observation.maxLength=1000;observation.rows=3;observation.style.width='100%';
  const confirmed=input('教练确认','checkbox');const preview=el('p','',form);
  function showPreview(){const values=Object.values(ratingInputs).map(i=>i.value),used=values.filter(Boolean).length;const total=used===5&&confirmed.checked?values.reduce((a,v)=>a+Number(v),0)*4:null;preview.textContent=`评价覆盖 ${used}/5 · ${total===null?'总分待确认':`教练评分 ${total}/100`} · 结果落点、深度与命中率另行评价`;}
  form.addEventListener('input',()=>{confirmed.checked=false;showPreview()});confirmed.addEventListener('input',e=>{e.stopPropagation();showPreview()});
  function reset(){selected=null;form.reset();start.value='';end.value='';confirmed.checked=false;showPreview();status.textContent='新评价 · 先圈定单次挥拍，再按五项标准评价';}
  function edit(shot){selected=shot.id;start.value=shot.start_frame;end.value=shot.end_frame;stroke.value=shot.stroke;goal.value=shot.goal;reviewer.value=shot.reviewer;observation.value=shot.observation;confirmed.checked=shot.confirmed;for(const [k,i] of Object.entries(machine))i.value=shot.machine[k]||'';for(const [k,i] of Object.entries(ratingInputs))i.value=shot.ratings[k]??'';showPreview();seek(shot.start_frame/state.fps);const stale=JSON.stringify(shot.evidence_sha256)!==JSON.stringify(state.evidence_sha256)||shot.score?.policy_sha256!==state.policy_sha256;if(stale){confirmed.checked=false;showPreview();}status.textContent=stale?'来源或标准已变化，请重新复核并确认':'已加载评价；修改后需重新确认';}
  function render(){rows.replaceChildren();for(const shot of state.document.shots){const stale=JSON.stringify(shot.evidence_sha256)!==JSON.stringify(state.evidence_sha256)||shot.score?.policy_sha256!==state.policy_sha256;const score=stale?null:shot.score?.score;button(`${shot.start_frame}–${shot.end_frame} 帧 · ${score==null?'草稿':`${score}/100 · 教练确认`}`,()=>edit(shot),rows);}if(!state.document.shots.length)el('p','暂无教练评分；不会根据拟合状态自动补分。',rows);}
  const save=button('保存评价',async()=>{
   save.disabled=true;
   try{
    if(start.value===''||end.value==='')throw Error('请先标记开始帧与结束帧');
    const shot={id:selected||crypto.randomUUID().replaceAll('-',''),start_frame:Number(start.value),end_frame:Number(end.value),stroke:stroke.value,goal:goal.value,reviewer:reviewer.value,observation:observation.value,confirmed:confirmed.checked,machine:Object.fromEntries(Object.entries(machine).map(([k,i])=>[k,i.value])),ratings:Object.fromEntries(Object.entries(ratingInputs).map(([k,i])=>[k,i.value===''?null:Number(i.value)])),pose_variant:document.getElementById('pose')?.value||'raw'};
    const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_sha256:state.video_sha256,policy_version:policy.version,policy_sha256:state.policy_sha256,base_revision:state.document.revision,evidence_sha256:state.evidence_sha256,shot})});const result=await response.json();if(!response.ok)throw Error(result.error||'保存失败');
    state={...state,...result};selected=shot.id;render();status.textContent=`已保存第 ${state.document.revision} 版 · ${shot.confirmed?'教练已确认':'草稿'} · 保留修订记录`;
   }catch(e){status.textContent=e.message}finally{save.disabled=false}
  });button('新增评价',reset);
  button('导出评价 JSON',()=>{const blob=new Blob([JSON.stringify(state.document,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='practice_review.json';a.click();URL.revokeObjectURL(a.href)});
  render();reset();
 }catch(e){status.textContent=e.message}
}
root.TennisPracticeCoach={mount};
})(typeof window==='undefined'?globalThis:window);
