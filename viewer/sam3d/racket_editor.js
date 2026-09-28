// Manual keyframes provide reviewable seeds for later racket segmentation/tracking.
(()=>{
'use strict';
const VIDEO_ID='30.56', SIZE=[1280,720], KEY='tennis-racket:30.56:1280x720:v1';
const PARTS=[
  ['handle_end','握柄末端'],
  ['throat','拍喉'],
  ['tip','拍头顶端'],
  ['rim_side','拍框侧缘'],
];
let frames=new Map(), editFrame=0, draft={}, selected=PARTS[0][0];
const status=message=>$('racketStatus').textContent=message;
function validPoint(p){return Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[0]<=SIZE[0]&&p[1]>=0&&p[1]<=SIZE[1]}
function validate(value){
  if(value?.video_id!==VIDEO_ID||value?.image_size?.[0]!==SIZE[0]||value?.image_size?.[1]!==SIZE[1])throw Error('视频或画面尺寸不匹配');
  if(value.fps!==25||!Array.isArray(value.frames))throw Error('需要 25 fps 的 frames 数组');
  const result=new Map();
  for(const row of value.frames){
    if(!Number.isInteger(row.frame)||row.frame<0||row.frame>=250||!row.points||typeof row.points!=='object')throw Error('标注帧号或点集无效');
    if(result.has(row.frame))throw Error('存在重复标注帧');
    const points={};
    for(const [name] of PARTS)if(row.points[name]!==undefined){if(!validPoint(row.points[name]))throw Error('无效点位：'+name);points[name]=row.points[name]}
    if(!Object.keys(points).length)throw Error('第 '+(row.frame+1)+' 帧没有标注点');
    result.set(row.frame,{frame:row.frame,points,source:row.source||'manual'});
  }
  return result;
}
function payload(){return{version:1,video_id:VIDEO_ID,fps:25,image_size:SIZE,coordinate_system:'original_video_pixels',frames:[...frames.values()].sort((a,b)=>a.frame-b.frame)}}
function persist(){
  try{localStorage.setItem(KEY,JSON.stringify(payload()));status('已保存 '+frames.size+' 个关键帧到本机；可导出 JSON');}
  catch(e){status('本机保存失败，请立即导出 JSON：'+e.message)}
}
try{const saved=localStorage.getItem(KEY);if(saved)frames=validate(JSON.parse(saved))}catch(e){frames=new Map()}
const section=document.createElement('details');section.id='racketAnnotations';
section.innerHTML=`<summary>球拍二维关键帧标注（第一阶段）</summary>
<p>在原视频中标记球拍，作为后续云端分割和三维重建的人工基准。当前仅显示二维标注，不代表已恢复球拍深度。</p>
<div class="row"><button id="racketOpen">标注当前帧</button><button id="racketExport">导出球拍 JSON</button><button id="racketImport">导入球拍 JSON</button><span id="racketStatus" role="status"></span></div>
<small>建议选择拍框清晰、不同挥拍角度的关键帧；至少标出握柄末端、拍喉、拍头顶端。拍框侧缘用于估计拍面方向。</small>`;
document.body.appendChild(section);
const dialog=document.createElement('dialog');dialog.id='racketEditor';
dialog.innerHTML=`<form method="dialog"><button class="close" aria-label="关闭球拍标注">×</button></form>
<h2>球拍关键帧标注</h2><p>点击画面给选中的点定位；方向键微调 1 像素，Shift 加方向键为 10 像素。球拍被遮挡的点可留空。</p>
<div class="row"><label>当前点 <select id="racketPart">${PARTS.map(([id,label])=>`<option value="${id}">${label}</option>`).join('')}</select></label>
<button id="racketClearPoint">清除当前点</button><button id="racketDeleteFrame">删除本帧标注</button></div>
<canvas id="racketImage" tabindex="0" aria-label="球拍标注画面"></canvas>
<p id="racketEditorStatus" role="status"></p>
<div class="row"><button id="racketSave">保存本帧</button><button id="racketCancel">取消</button></div>`;
document.body.appendChild(dialog);
const style=document.createElement('style');
style.textContent=`#racketEditor{width:min(1180px,96vw);max-height:94vh;overflow:auto;background:#152233;color:#e6eef8;border:1px solid #5b7491;border-radius:16px;padding:20px}#racketEditor::backdrop{background:#000b}#racketEditor .close{float:right}#racketEditor h2{margin:0}#racketImage{display:block;width:100%;height:auto;max-height:65vh;object-fit:contain;background:#0b111b;cursor:crosshair;outline:1px solid #51647a}#racketImage:focus{outline:2px solid #6fcaff}#racketEditorStatus{min-height:24px}`;
document.head.appendChild(style);
const canvas=$('racketImage'),g=canvas.getContext('2d'),still=document.createElement('canvas');
canvas.width=still.width=SIZE[0];canvas.height=still.height=SIZE[1];
function painted(ctx,points,sx,sy){
  const a=points.handle_end,b=points.throat,c=points.tip;
  ctx.save();ctx.lineWidth=3;ctx.strokeStyle='#fbd36d';ctx.setLineDash([8,5]);
  if(a&&b){ctx.beginPath();ctx.moveTo(a[0]*sx,a[1]*sy);ctx.lineTo(b[0]*sx,b[1]*sy);ctx.stroke()}
  if(b&&c){ctx.beginPath();ctx.moveTo(b[0]*sx,b[1]*sy);ctx.lineTo(c[0]*sx,c[1]*sy);ctx.stroke()}
  ctx.setLineDash([]);
  for(const [name,label] of PARTS){const p=points[name];if(!p)continue;const x=p[0]*sx,y=p[1]*sy;ctx.fillStyle=name===selected?'#5de5fb':'#fbd36d';ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fill();ctx.fillStyle='#101824';ctx.font='bold 16px system-ui';ctx.fillText(label,x+10,y-9)}
  ctx.restore();
}
function draw(){
  g.clearRect(0,0,...SIZE);g.drawImage(still,0,0);painted(g,draft,1,1);
  $('racketEditorStatus').textContent=`第 ${editFrame+1}/250 帧 · 已标 ${Object.keys(draft).length}/4 点 · 当前：${PARTS.find(x=>x[0]===selected)[1]}`;
  $('racketPart').value=selected;
}
function open(){
  if(!data||v.readyState<2){status('等待视频和姿态数据加载后再标注');return}
  v.pause();editFrame=n;draft=structuredClone(frames.get(n)?.points||{});selected=PARTS[0][0];
  still.getContext('2d').drawImage(v,0,0,...SIZE);dialog.showModal();draw();canvas.focus();
}
$('racketOpen').onclick=open;
$('racketPart').onchange=e=>{selected=e.target.value;draw();canvas.focus()};
canvas.onclick=e=>{
  const rect=canvas.getBoundingClientRect(),scale=Math.min(rect.width/SIZE[0],rect.height/SIZE[1]);
  const left=rect.left+(rect.width-SIZE[0]*scale)/2,top=rect.top+(rect.height-SIZE[1]*scale)/2;
  const point=[(e.clientX-left)/scale,(e.clientY-top)/scale];
  if(!validPoint(point))return;
  draft[selected]=point;
  selected=PARTS.find(([name])=>!draft[name])?.[0]||selected;
  draw();canvas.focus();
};
canvas.onkeydown=e=>{
  const move={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
  if(!move||!draft[selected])return;
  e.preventDefault();draft[selected]=draft[selected].map((x,i)=>Math.max(0,Math.min(SIZE[i],x+move[i]*(e.shiftKey?10:1))));draw();
};
$('racketClearPoint').onclick=()=>{delete draft[selected];draw();canvas.focus()};
$('racketDeleteFrame').onclick=()=>{frames.delete(editFrame);persist();render();dialog.close()};
$('racketCancel').onclick=()=>dialog.close();
$('racketSave').onclick=()=>{
  if(!Object.keys(draft).length){$('racketEditorStatus').textContent='请至少标记一个可见点';return}
  frames.set(editFrame,{frame:editFrame,points:structuredClone(draft),source:'manual'});
  persist();render();dialog.close();
};
$('racketExport').onclick=()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload(),null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='racket_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
const file=document.createElement('input');file.type='file';file.accept='.json,application/json';file.hidden=true;document.body.appendChild(file);
$('racketImport').onclick=()=>file.click();
file.onchange=async()=>{
  try{if(!file.files?.[0])return;frames=validate(JSON.parse(await file.files[0].text()));persist();render()}
  catch(e){status('导入失败：'+e.message)}
  finally{file.value=''}
};
window.renderRacketOverlay=(ctx,rect,frame)=>{
  const row=frames.get(frame);if(row)painted(ctx,row.points,rect.width/SIZE[0],rect.height/SIZE[1]);
};
status('本机已有 '+frames.size+' 个球拍关键帧');
})();
