/* Evidence-linked coaching. Reports are bound to the source video, never reused across uploads. */
(function(root){
function validate(report,meta){
 if(report.version!==1||!meta.video_sha256||report.video_sha256!==meta.video_sha256)throw Error('分析与当前视频不匹配，请为本视频重新分析');
 const duration=meta.frames/meta.fps;
 if(!Number.isFinite(duration)||duration<=0||!Array.isArray(report.shots)||!report.shots.length)throw Error('分析结构无效');
 for(const shot of report.shots){if(!Array.isArray(shot.phases)||!shot.phases.length)throw Error('缺少动作阶段');let previous=-1;
  for(const p of shot.phases){if(!Number.isFinite(p.start)||!Number.isFinite(p.end)||p.start<0||p.start<previous||p.end<=p.start||p.end>duration+.001||typeof p.observation!=='string')throw Error('动作时间超出视频或顺序错误');previous=p.start;}
 }
 return report;
}
async function mount({container,before,video,seek,reportUrl='coaching_report.json',metaUrl='mesh_meta.json'}){
 const panel=document.createElement('section');panel.className='coach-panel';if(before)before.before(panel);else container.append(panel);
 const style=document.createElement('style');style.textContent='.coach-panel{margin:16px 0;padding:18px;border:1px solid #405c79;border-radius:12px;background:#192638;color:#e6eef8}.coach-panel h2{margin:0 0 10px}.coach-panel button{font:inherit;background:#2d4560;color:#e6eef8;border:1px solid #587894;border-radius:7px;padding:8px;margin:4px;cursor:pointer}.coach-panel button[aria-pressed=true]{background:#155a72;border-color:#96deef}.coach-panel p{line-height:1.6}.coach-panel .coach-cue{padding:12px;border-left:3px solid #82d6bb;background:#153b3a}.coach-panel small{color:#b6c9de}';document.head.append(style);
 const el=(tag,text,parent=panel)=>{const n=document.createElement(tag);n.textContent=text;parent.append(n);return n};
 el('h2','◎ 动作指导');
 try{
  const response=await fetch(reportUrl);if(response.status===404){el('p','本视频尚无教学分析。生成姿态后，还需结合原视频确认击球阶段；不会套用默认视频的指导。');return}if(!response.ok)throw Error('教学分析加载失败');
  const mr=await fetch(metaUrl);if(!mr.ok)throw Error('视频元数据加载失败');const meta=await mr.json(),report=validate(await response.json(),meta);
  el('small','草稿 · 待复核');const cue=el('p',report.cue);cue.className='coach-cue';
  const shots=el('div',''),phases=el('div',''),evidence=el('p',''),now=el('p','');evidence.setAttribute('aria-live','polite');
  const stopLabel=el('label','');const stop=document.createElement('input');stop.type='checkbox';stop.checked=true;stopLabel.append(stop,document.createTextNode(' 选定片段播放到末尾时暂停'));
  let selected=null;const phaseButtons=[];const shotButtons=[];
  const choosePhase=p=>{selected=p;seek(p.start);evidence.textContent=`${p.label} · ${p.start.toFixed(2)}–${p.end.toFixed(2)} 秒：${p.observation} 证据：${p.evidence}。${p.confidence}。`;for(const [b,x] of phaseButtons)b.setAttribute('aria-pressed',String(x===p));};
  function chooseShot(shot){phases.replaceChildren();phaseButtons.length=0;selected=null;for(const [b,x] of shotButtons)b.setAttribute('aria-pressed',String(x===shot));for(const p of shot.phases){const b=el('button',`${p.label} ${p.start.toFixed(2)}s`,phases);b.onclick=()=>choosePhase(p);phaseButtons.push([b,p]);}evidence.textContent='点击阶段定位视频与 3D，再使用播放按钮查看连续动作。';}
  for(const shot of report.shots){const b=el('button',shot.label,shots);b.onclick=()=>chooseShot(shot);shotButtons.push([b,shot]);}chooseShot(report.shots[0]);
  video.addEventListener('timeupdate',()=>{const t=video.currentTime;const matches=report.shots.flatMap(s=>s.phases.map(p=>({s,p}))).filter(({p})=>t>=p.start&&t<p.end);now.textContent=matches.length?`当前：${matches[0].s.label} / ${matches[0].p.label}`:'当前未标注动作阶段';if(selected&&stop.checked&&!video.paused&&t>=selected.end){const end=selected.end;selected=null;seek(Math.min(end,Math.max(0,(meta.frames-1)/meta.fps)));}});
  const details=el('details','');el('summary','分析来源与使用限制',details);el('p',report.summary,details);el('p',report.cue_evidence,details);el('p',report.provenance,details);for(const text of report.limitations||[])el('p',text,details);
 }catch(e){el('p',e.message);}
}
root.TennisCoaching={validate,mount};
})(typeof window==='undefined'?globalThis:window);
