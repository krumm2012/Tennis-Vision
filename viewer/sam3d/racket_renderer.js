// Rigid racket in the same WebGL depth buffer and display coordinates as SAM body.
(()=>{
'use strict';
const ID='30.56', SIZE=[1280,720], STORAGE='tennis-racket-poses:30.56:v1';
const PARTS=['handle_end','throat','tip','rim_side','rim_opposite'];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,s)=>a.map(x=>x*s);
const unit=a=>{const length=Math.hypot(...a);return length?mul(a,1/length):[0,0,1]};
function geometry(model){
 const vertices=[];
 function triangle(a,b,c,color){for(const p of [a,b,c])vertices.push(...p,...color)}
 function rod(a,b,r,color,sides=8){
  const axis=unit(sub(b,a)),reference=Math.abs(axis[2])<.8?[0,0,1]:[1,0,0],u=unit(cross(axis,reference)),v=unit(cross(axis,u));
  for(let k=0;k<sides;k++){
   const angle=k*2*Math.PI/sides,next=(k+1)*2*Math.PI/sides;
   const offset=t=>add(mul(u,Math.cos(t)*r),mul(v,Math.sin(t)*r));
   const A=add(a,offset(angle)),B=add(a,offset(next)),C=add(b,offset(next)),D=add(b,offset(angle));
   triangle(A,B,C,color);triangle(A,C,D,color);
  }
 }
 const dark=[.14,.20,.28],grip=[.24,.29,.33],accent=[.72,.25,.18],string=[.70,.75,.74];
 const length=model.length_m,center=model.head_center_y_m,half=length-center,width=model.head_half_width_m;
 rod([0,0,0],[0,.21,0],.014,grip,12);
 rod([0,.20,0],[0,.33,0],.008,dark,10);
 rod([0,.24,0],[0,.245,0],.015,accent,12);
 const segments=40;
 const rim=t=>[width*Math.sin(t),center+half*Math.cos(t),0];
 for(let k=0;k<segments;k++)rod(rim(k*2*Math.PI/segments),rim((k+1)*2*Math.PI/segments),.008,dark,7);
 rod([0,.315,0],[-.059,center-half*.89,0],.007,dark);
 rod([0,.315,0],[.059,center-half*.89,0],.007,dark);
 // Separate shallow string mesh makes the racket-plane orientation legible in 3D.
 for(let k=-5;k<=5;k++){
  const x=k*width/6,extent=half*Math.sqrt(Math.max(0,1-x*x/(width*width)))*.94;
  rod([x,center-extent,.002],[x,center+extent,.002],.0007,string,4);
 }
 for(let k=-7;k<=7;k++){
  const y=center+k*half/8,extent=width*Math.sqrt(Math.max(0,1-(y-center)**2/(half*half)))*.94;
  rod([-extent,y,-.002],[extent,y,-.002],.0007,string,4);
 }
 return new Float32Array(vertices);
}
function validPose(json){
 if(json?.version!==1||json.video_id!==ID||json.fps!==25||JSON.stringify(json.image_size)!==JSON.stringify(SIZE)||!Array.isArray(json.frames))throw Error('球拍姿态文件与当前视频不匹配');
 const model=json.model;
 for(const key of ['length_m','head_half_width_m','head_center_y_m','throat_y_m','grip_y_m'])if(!Number.isFinite(model?.[key]))throw Error('球拍模型尺寸无效：'+key);
 const frames=new Map();
 for(const row of json.frames){
  if(!Number.isInteger(row.frame)||row.frame<0||row.frame>=250||frames.has(row.frame))throw Error('球拍帧号重复或无效');
  if(row.status==='fitted'){
   if(!Array.isArray(row.translation_camera_m)||row.translation_camera_m.length!==3||!row.translation_camera_m.every(Number.isFinite))throw Error('球拍平移无效');
   if(!Array.isArray(row.rotation_camera_columns)||row.rotation_camera_columns.length!==3||row.rotation_camera_columns.some(r=>!Array.isArray(r)||r.length!==3||!r.every(Number.isFinite)))throw Error('球拍旋转无效');
  }
  if(row.joint_hand_camera&&(!Array.isArray(row.joint_hand_camera)||row.joint_hand_camera.length!==21||row.joint_hand_camera.some(p=>!Array.isArray(p)||p.length!==3||!p.every(Number.isFinite))))throw Error('联合手部关节无效');
  frames.set(row.frame,row);
 }
 return{frames,model,summary:json.summary};
}
class RacketLayer{
 constructor(){this.pose=null;this.glState=new WeakMap();this.enabled=true;}
 load(json){this.pose=validPose(json);this.grasp=json.grasp_calibration;this.assetMesh=null;this.glState=new WeakMap();this.updateStatus();render();
  if(json.model.asset_file==='wilson_mesh.bin'){const loaded=this.pose;fetch('wilson_mesh.bin').then(r=>{if(!r.ok)throw Error('Wilson模型加载失败');return r.arrayBuffer()}).then(b=>{if(this.pose!==loaded)return;this.assetMesh=new Float32Array(b);this.glState=new WeakMap();render()}).catch(e=>{document.getElementById('racket3dStatus').textContent=e.message})}}

 updateStatus(frame=n){
  const status=document.getElementById('racket3dStatus');if(!status)return;
  if(!this.pose){status.textContent='尚无球拍三维姿态；请导入拟合 JSON';return}
  const total=[...this.pose.frames.values()].filter(x=>x.status==='fitted').length,row=this.pose.frames.get(frame);
  if(!row){status.textContent=`已拟合 ${total} 帧；当前第 ${frame+1} 帧无可靠球拍姿态`;return}
  if(row.status!=='fitted'){status.textContent=`第 ${frame+1} 帧：${row.status}；球拍不显示`;return}
  if(row.source==='joint_sequence_candidate'){status.textContent=`第 ${frame+1} 帧 · 联合候选 · 轮廓残差 ${row.mask_fit_rms_px.toFixed(1)} px · 接触先验残差 ${row.contact_proxy_mm.toFixed(1)} mm · ${row.review_reasons.length?'需要复核':'未触发阈值，非准确性验证'} · ${row.hand_mesh_updated?'手指网格近似变形':'手指网格未更新'}`;return}
  const warning=row.ambiguous?' · 单目拍面朝向有歧义':'';
  const rough=row.source?.includes('estimate')?' · 手工粗标待复核':'';
  const method=row.quality==='temporal_estimate'?'掌柄与时序约束估计（需复核）':row.quality==='interpolated'?'邻帧插值（需复核）':row.quality==='silhouette_fitted'?'轮廓拟合':'关键点拟合';
  const error=Number.isFinite(row.mask_fit_rms_px)?`轮廓误差 ${row.mask_fit_rms_px} px`:`二维重投影 ${row.real_reprojection_rms_px??'未知'} px`;
  const confidence=Number.isFinite(row.detection_confidence)?` · 检测分数 ${row.detection_confidence.toFixed(3)}`:'';
  const summary=this.pose.summary?`；全片 ${this.pose.summary.silhouette_fitted} 帧轮廓拟合 / ${this.pose.summary.constrained_estimate??this.pose.summary.interpolated} 帧${this.pose.summary.constrained_estimate!=null?'约束估计':'插值'}`:`；已拟合 ${total} 帧`;
  status.textContent=`第 ${frame+1} 帧：${method} · ${error}${confidence} · 握拍点至手腕 ${Number(row.wrist_gap_m).toFixed(3)} m${warning}${rough}${Number.isFinite(row.grip_axis_error_deg)?` · 掌柄方向偏差 ${row.grip_axis_error_deg.toFixed(0)}°${row.review_reasons?.includes('hand_axis_conflict')?'（手部需复核）':''}`:''}${this.grasp?' · 掌内握点约束（标定待确认）':''}${summary}`;
 }
 init(gl,model){
  const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s};
  const vs=`#version 300 es
  precision highp float;layout(location=0) in vec3 position;layout(location=1) in vec3 vertexColor;
  uniform mat3 racketRotation;uniform vec3 racketTranslation;uniform vec3 sourceRoot;
  uniform mat3 basis;uniform vec3 shift;uniform vec3 center;uniform vec2 viewport;uniform vec2 angles;uniform float scale;
  out vec3 rgb;
  void main(){vec3 camera=racketRotation*position+racketTranslation;vec3 p=basis*(camera-sourceRoot)+shift-center;
   float cy=cos(angles.x),sy=sin(angles.x),cp=cos(angles.y),sp=sin(angles.y);
   float x=p.x*cy+p.z*sy,z=-p.x*sy+p.z*cy,y=p.y*cp-z*sp,depth=p.y*sp+z*cp;
   gl_Position=vec4(x*scale*2./viewport.x,y*scale*2./viewport.y,-depth/20.,1.);
   rgb=vertexColor;
  }`;
  const fs=`#version 300 es
  precision highp float;in vec3 rgb;uniform float warning;out vec4 color;
  void main(){color=vec4(mix(rgb,vec3(.96,.63,.17),warning*.65),1.);}`;
  const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  const vao=gl.createVertexArray(),buffer=gl.createBuffer(),mesh=this.assetMesh||geometry(model);gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,mesh,gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);gl.bindVertexArray(null);
  const uniforms={};for(const name of ['racketRotation','racketTranslation','sourceRoot','basis','shift','center','viewport','angles','scale','warning'])uniforms[name]=gl.getUniformLocation(program,name);
  return{program,vao,uniforms,count:mesh.length/6};
 }
 draw(renderer,frame,mode,basis,shift,center,angles,scale,rect){
  if(!this.enabled||!this.pose)return;
  const row=this.pose.frames.get(frame);if(row?.status!=='fitted')return;
  const gl=renderer.gl;let state=this.glState.get(gl);if(!state){state=this.init(gl,this.pose.model);this.glState.set(gl,state)}
  const u=state.uniforms,sourceRoot=renderer.meta.source_roots[frame];
  const joints=data?.frames?.[frame],rawWrist=joints?.raw?.[41],displayWrist=joints?.[mode]?.[41];
  const correction=rawWrist&&displayWrist?sub(displayWrist,rawWrist):[0,0,0];
  let translation=add(row.translation_camera_m,correction),rotation=row.rotation_camera_columns;
  if(this.grasp&&joints){
   const palm=p=>{const w=p[41],forward=unit(sub(mul(add(add(p[28],p[32]),add(p[36],p[40])),.25),w));let across=sub(p[28],p[40]);across=unit(sub(across,mul(forward,across.reduce((a,x,i)=>a+x*forward[i],0))));return [across,forward,cross(across,forward)]};
   const raw=palm(joints.raw),shown=palm(joints[mode]);
   const rotateHand=v=>shown.reduce((sum,axis,k)=>add(sum,mul(axis,raw[k].reduce((a,x,j)=>a+x*v[j],0))),[0,0,0]);
   const columns=[0,1,2].map(k=>rotateHand(rotation.map(r=>r[k])));rotation=[0,1,2].map(j=>columns.map(col=>col[j]));
   const offset=row.palm_offset_m||this.grasp.palm_offset_m;const target=add(add(sourceRoot,displayWrist),shown.reduce((sum,axis,k)=>add(sum,mul(axis,offset[k])),[0,0,0]));
   translation=sub(target,mul(rotation.map(r=>r[1]),this.pose.model.grip_y_m));
  }
  // cv2 stores row-major matrices; WebGL uniformMatrix3fv expects columns.
  const R=rotation,columns=[0,1,2].flatMap(col=>R.map(line=>line[col]));
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,renderer.canvas.width,renderer.canvas.height);
  gl.useProgram(state.program);gl.bindVertexArray(state.vao);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);
  gl.uniformMatrix3fv(u.racketRotation,false,columns);gl.uniform3fv(u.racketTranslation,translation);gl.uniform3fv(u.sourceRoot,sourceRoot);
  gl.uniformMatrix3fv(u.basis,false,basis.flat());gl.uniform3fv(u.shift,shift);gl.uniform3fv(u.center,center);
  gl.uniform2fv(u.viewport,[rect.width,rect.height]);gl.uniform2fv(u.angles,angles);gl.uniform1f(u.scale,scale);
  gl.uniform1f(u.warning,this.assetMesh?0:(row.ambiguous||row.source?.includes('estimate')?1:0));gl.drawArrays(gl.TRIANGLES,0,state.count);
  if(row.joint_hand_camera&&document.getElementById('jointHandVisible')?.checked){
   if(!state.hand){const vao=gl.createVertexArray(),buffer=gl.createBuffer();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);state.hand={vao,buffer,frame:-1}}
   const h=state.hand;gl.bindVertexArray(h.vao);gl.bindBuffer(gl.ARRAY_BUFFER,h.buffer);
   if(h.frame!==frame){const lines=[];for(const tip of [0,4,8,12,16]){const chain=[20,tip+3,tip+2,tip+1,tip];for(let k=1;k<chain.length;k++)for(const id of [chain[k-1],chain[k]])lines.push(...row.joint_hand_camera[id],1,.65,.25)}gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(lines),gl.DYNAMIC_DRAW);h.count=lines.length/6;h.frame=frame}
   gl.uniformMatrix3fv(u.racketRotation,false,[1,0,0,0,1,0,0,0,1]);gl.uniform3fv(u.racketTranslation,[0,0,0]);gl.uniform1f(u.warning,0);gl.disable(gl.DEPTH_TEST);gl.drawArrays(gl.LINES,0,h.count);gl.enable(gl.DEPTH_TEST);
  }
 }
 overlay(ctx,rect,frame){
  const row=this.pose?.frames.get(frame);if(!row||row.status!=='fitted')return;
  ctx.save();ctx.lineWidth=1.5;ctx.strokeStyle=row.ambiguous?'#ffbd62':'#69f0a8';
  if(row.projected_head_outline?.length){ctx.strokeStyle=row.quality!=='silhouette_fitted'?'#ff8e6e':'#69f0a8';ctx.beginPath();row.projected_head_outline.forEach((p,i)=>{const x=p[0]*rect.width/SIZE[0],y=p[1]*rect.height/SIZE[1];i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.stroke()}
  for(const name of PARTS){const p=row.projected_points?.[name];if(!p)continue;const x=p[0]*rect.width/SIZE[0],y=p[1]*rect.height/SIZE[1];ctx.beginPath();ctx.moveTo(x-4,y-4);ctx.lineTo(x+4,y+4);ctx.moveTo(x-4,y+4);ctx.lineTo(x+4,y-4);ctx.stroke()}
  if(row.joint_hand_image&&document.getElementById('jointHandVisible')?.checked){ctx.strokeStyle='#ffb665';for(const tip of [0,4,8,12,16]){const chain=[20,tip+3,tip+2,tip+1,tip];ctx.beginPath();chain.forEach((id,k)=>{const p=row.joint_hand_image[id],x=p[0]*rect.width/SIZE[0],y=p[1]*rect.height/SIZE[1];k?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}}
  ctx.restore();
 }
}
const layer=new RacketLayer();
const section=document.createElement('details');section.id='racket3dPanel';section.innerHTML=`<summary>球拍数据与高级操作</summary>
<p>Wilson 版使用真实拍框、掌内握持位置、镜中候选观测和时序约束。绿色轮廓通过拟合筛选，橙色为待复核估计。掌内握点及实物尺寸尚待人工确认，拍面正反方向仍有歧义。可加载上一版比较。</p>
<div class="row"><label><input id="racket3dVisible" type="checkbox" checked>显示三维球拍</label><button id="racket3dImport">导入拟合 JSON</button><button id="racket3dReview">下一需复核帧</button><button id="racket3dReload">加载服务器最新版</button><button id="racket3dGrip">握持修正版（待复核）</button><button id="racket3dPrevious">上一版对比</button><a id="racketJointFull" href="joint_fit_v4/full/viewer.html" hidden>完整 Viewer · 联合候选</a><a id="racketJointDiagnostic" href="joint_fit_v4/viewer.html" hidden>全片掌柄联合诊断</a><span id="racket3dStatus" role="status"></span></div>`;
(document.getElementById('toolPanels')||document.body).appendChild(section);
if(window.TENNIS_JOINT_CANDIDATE){section.querySelector('p').textContent=window.TENNIS_HAND_MESH_PREVIEW?'原片人体和联合球拍使用同一相机坐标。手指网格按优化关节近似变形；橙色线显示目标骨架，网格接触尚未验证。':'原片人体和联合球拍使用同一相机坐标。橙色手部关节为优化结果，透视叠加便于查看，尚未驱动手指网格。';const label=document.createElement('label');label.innerHTML='<input id="jointHandVisible" type="checkbox" checked>显示优化手部关节';section.querySelector('.row').prepend(label);label.querySelector('input').onchange=()=>render();for(const id of ['racket3dImport','racket3dGrip','racket3dPrevious'])document.getElementById(id).hidden=true;}
const quick=document.createElement('div');quick.id='racketQuick';quick.className='quickbar';quick.setAttribute('role','group');quick.setAttribute('aria-label','球拍快捷操作');quick.innerHTML='<strong>球拍</strong>';document.querySelector('nav.playback')?.after(quick);quick.append(document.getElementById('racket3dVisible').closest('label'));if(window.TENNIS_JOINT_CANDIDATE)quick.append(document.getElementById('jointHandVisible').closest('label'));quick.append(document.getElementById('racket3dReview'),document.getElementById('racket3dStatus'));
const input=document.createElement('input');input.type='file';input.accept='.json,application/json';input.hidden=true;document.body.appendChild(input);
document.getElementById('racket3dVisible').onchange=e=>{layer.enabled=e.target.checked;render()};
document.getElementById('racket3dImport').onclick=()=>input.click();
document.getElementById('racket3dReview').onclick=()=>{const rows=[...(layer.pose?.frames.values()||[])].filter(r=>r.review_reasons?.length||r.quality==='interpolated'||r.mask_fit_rms_px>3||r.detection_confidence<.03);const target=rows.find(r=>r.frame>n)||rows[0];if(target)jump(target.frame)};
document.getElementById('racket3dGrip').onclick=async()=>{try{const r=await fetch('racket_poses_v3.json',{cache:'no-store'});if(!r.ok)throw Error('握持修正版不可用');layer.load(await r.json())}catch(e){document.getElementById('racket3dStatus').textContent=e.message}};
if(!window.TENNIS_JOINT_CANDIDATE)fetch('joint_fit_v4/full/mesh_meta.json',{cache:'no-store'}).then(r=>{if(r.ok)document.getElementById('racketJointFull').hidden=false}).catch(()=>{});
fetch('joint_fit_v4/report.json',{cache:'no-store'}).then(r=>{if(r.ok)document.getElementById('racketJointDiagnostic').hidden=false}).catch(()=>{});
document.getElementById('racket3dPrevious').onclick=async()=>{try{const r=await fetch('racket_poses_v1.json',{cache:'no-store'});if(!r.ok)throw Error('上一版文件不可用');layer.load(await r.json())}catch(e){document.getElementById('racket3dStatus').textContent=e.message}};
document.getElementById('racket3dReload').onclick=async()=>{try{const response=await fetch('racket_poses.json',{cache:'no-store'});if(!response.ok)throw Error('姿态文件加载失败');layer.load(await response.json());if(!window.TENNIS_JOINT_CANDIDATE)localStorage.removeItem(STORAGE)}catch(e){document.getElementById('racket3dStatus').textContent=e.message}};
input.onchange=async()=>{try{if(!input.files?.[0])return;const content=await input.files[0].text();layer.load(JSON.parse(content));localStorage.setItem(STORAGE,content)}catch(e){document.getElementById('racket3dStatus').textContent='导入失败：'+e.message}finally{input.value=''}};
try{const stored=window.TENNIS_JOINT_CANDIDATE?null:localStorage.getItem(STORAGE);if(stored)layer.load(JSON.parse(stored))}catch(e){localStorage.removeItem(STORAGE)}
const previousDraw=SamMeshRenderer.prototype.draw;
SamMeshRenderer.prototype.draw=function(...args){const drawn=previousDraw.apply(this,args);if(this.ready){if(this.video.readyState>=2)layer.draw(this,...args);layer.updateStatus(args[0])}return drawn};
const previousOverlay=window.renderRacketOverlay;
window.renderRacketOverlay=(ctx,rect,frame)=>{previousOverlay?.(ctx,rect,frame);layer.overlay(ctx,rect,frame)};
layer.updateStatus();
fetch('racket_poses.json').then(r=>r.ok?r.json():null).then(json=>{if(json&&!layer.pose)layer.load(json)}).catch(()=>{});
})();
