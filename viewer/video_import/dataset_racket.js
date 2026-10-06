// Current-video Wilson poses share the SAM display transform and camera uniforms.
class DatasetRacketLayer {
 constructor(){this.enabled=true;this.rows=[];this.state=null;}
 async load(renderer,poseFile='racket_poses.json'){
  const response=await fetch(poseFile,{cache:'no-store'});if(!response.ok)return;
  const data=await response.json(),meta=renderer.meta;
  if(data.video_sha256!==meta.video_sha256||data.fps!==meta.fps||JSON.stringify(data.image_size)!==JSON.stringify(meta.image_size)||data.frames.length!==meta.frames)throw Error('球拍数据与当前视频不匹配');
  this.poseFile=poseFile;this.meta=meta;this.rows=data.frames;this.summary=data.summary;this.model=data.model;this.fps=data.fps;this.lastFrame=-1;
  try{const response=await fetch('racket_keypoints.json',{cache:'no-store'});if(response.ok){const evidence=await response.json();if(evidence.video_sha256===meta.video_sha256&&evidence.frames.length===meta.frames)this.observations=evidence.frames}}catch{}
  this.transforms=this.rows.map((row,i)=>{if(row.status!=='fitted')return null;const r=row.rotation_camera_columns,m=new THREE.Matrix4().set(...r[0],0,...r[1],0,...r[2],0,0,0,0,1),q=new THREE.Quaternion().setFromRotationMatrix(m);return {q,grip:new THREE.Vector3().fromArray(row.grip_target_camera_m||new THREE.Vector3().fromArray(row.translation_camera_m).add(new THREE.Vector3(0,this.model.grip_y_m,0).applyQuaternion(q)).toArray()).sub(new THREE.Vector3().fromArray(meta.source_roots[i]))}});this.tempQuaternion=new THREE.Quaternion();this.tempMatrix=new THREE.Matrix4();this.localGrip=new THREE.Vector3(0,this.model.grip_y_m,0);
  const responseMesh=await fetch(['wilson_mesh_directional.bin','wilson_mesh_reference.bin'].includes(data.model.asset_file)?data.model.asset_file:'wilson_mesh.bin');if(!responseMesh.ok)throw Error('球拍模型加载失败');
  const values=new Float32Array(await responseMesh.arrayBuffer());if(!values.length||values.length%18||!values.every(Number.isFinite))throw Error('球拍模型无效');
  const buffer=new THREE.InterleavedBuffer(values,6),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));geometry.setAttribute('vertexColor',new THREE.InterleavedBufferAttribute(buffer,3,3));
  const uniforms={};for(const name of ['sourceRoot','basis','shift'])uniforms[name]=renderer.u[name];Object.assign(uniforms,{rotation:{value:new THREE.Matrix3()},translation:{value:new THREE.Vector3()},estimated:{value:0}});
  const material=new THREE.RawShaderMaterial({glslVersion:THREE.GLSL3,side:THREE.DoubleSide,uniforms,vertexShader:`precision highp float;in vec3 position;in vec3 vertexColor;uniform mat4 projectionMatrix,viewMatrix;uniform mat3 rotation,basis;uniform vec3 translation,sourceRoot,shift;out vec3 rgb;void main(){vec3 p=rotation*position+translation;gl_Position=projectionMatrix*viewMatrix*vec4(basis*(p-sourceRoot)+shift,1.);rgb=vertexColor;}`,fragmentShader:`precision highp float;in vec3 rgb;uniform float estimated;out vec4 color;void main(){color=vec4(mix(rgb,vec3(1.,.65,.22),estimated*.25),1.);}`});
  const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;mesh.visible=false;mesh.renderOrder=1;renderer.scene.add(mesh);this.state={mesh,uniforms};renderer.addLayer(this);

 }
 update(renderer,n,mode){
  if(!this.state)return;const row=this.rows[n];const warning=document.getElementById('fitWarning');if(warning)warning.textContent=this.enabled&&row?.grip_axis_error_deg>35?`⚠ 拍柄方向待复核 · 手部估计夹角 ${row.grip_axis_error_deg.toFixed(0)}°${row.quality!=='silhouette_fitted'?' · 当前为估计帧':''}`:'';const label=document.getElementById('gripStatus');if(label)label.textContent=row?.status==='fitted'&&row.face_angle_to_camera_deg!==undefined?`柄底→握点 ${(this.model.grip_y_m*100).toFixed(2)}cm · 法线/相机 ${row.face_angle_to_camera_deg.toFixed(0)}° · ${row.quality==='silhouette_fitted'?'轮廓支持':'约束估计'}${row.stereo_shaft_used?' · 双视角拍柄':''}${row.physical_face_sign_verified?' · A侧已校准':' · 正负待校准'}`:'';this.state.mesh.visible=this.enabled&&row?.status==='fitted';if(!this.state.mesh.visible)return;
  this.lastFrame=n;this.present(renderer,n);
 }
 present(renderer,n){
  const current=this.transforms[n];if(!current)return;
  // The body uses decoded frames. Advancing only the racket toward video.currentTime
  // separates the grip from the fingers during playback, especially under load.
  this.tempQuaternion.copy(current.q);this.tempMatrix.makeRotationFromQuaternion(current.q);this.state.uniforms.rotation.value.setFromMatrix4(this.tempMatrix);
  this.state.uniforms.translation.value.copy(current.grip).sub(this.localGrip.clone().applyQuaternion(current.q)).add(renderer.u.sourceRoot.value);
  this.state.uniforms.estimated.value=this.rows[n].quality==='silhouette_fitted'?0:1;
 }
 dispose(){cancelAnimationFrame(this.animation);this.state?.mesh.geometry.dispose();this.state?.mesh.material.dispose();}
 overlay(context,transform,n){
  const row=this.rows[n];if(!this.enabled||row?.status!=='fitted')return;context.strokeStyle=row.quality!=='silhouette_fitted'?'#ffbd62':'#6ce5a3';context.lineWidth=1.5;context.beginPath();row.projected_head_outline?.forEach((p,i)=>{const q=transform(p);i?context.lineTo(...q):context.moveTo(...q)});context.closePath();context.stroke();
  if(this.showEvidence){
   const labels={handle_end:'柄底',throat:'线床下缘',tip:'头',head_center:'心',rim_side:'A',rim_opposite:'B',grip_center:'握点'};
   for(const [name,p] of Object.entries(row.observed_keypoints||this.observations?.[n]?.real?.points||{})){const q=transform(p);context.fillStyle='#63e5db';context.beginPath();context.arc(...q,3,0,Math.PI*2);context.fill();context.font='11px system-ui';context.fillText(labels[name]||name,q[0]+5,q[1]-5)}
   let arrow=row.projected_face_normal;
   if(!arrow&&this.meta){const R=row.rotation_camera_columns,t=row.translation_camera_m,head=t.map((v,i)=>v+R[i][1]*this.model.head_center_y_m),tip=head.map((v,i)=>v+R[i][2]*.15),project=p=>[p[0]/p[2]*this.meta.focal[n]+this.meta.image_size[0]/2,p[1]/p[2]*this.meta.focal[n]+this.meta.image_size[1]/2];if(head[2]>0&&tip[2]>0)arrow=[project(head),project(tip)]}
   if(arrow){const [a,b]=arrow.map(transform),angle=Math.atan2(b[1]-a[1],b[0]-a[0]);context.strokeStyle=row.physical_face_sign_verified?'#63e5db':'#ffbd62';context.setLineDash(row.physical_face_sign_verified?[]:[4,3]);context.beginPath();context.moveTo(...a);context.lineTo(...b);context.moveTo(b[0]-7*Math.cos(angle-.4),b[1]-7*Math.sin(angle-.4));context.lineTo(...b);context.lineTo(b[0]-7*Math.cos(angle+.4),b[1]-7*Math.sin(angle+.4));context.stroke();context.setLineDash([])}
  }
 }
}
