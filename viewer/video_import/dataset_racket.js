// Current-video Wilson poses share the SAM display transform and camera uniforms.
class DatasetRacketLayer {
 constructor(){this.enabled=true;this.rows=[];this.state=null;}
 async load(renderer){
  const response=await fetch('racket_poses.json',{cache:'no-store'});if(!response.ok)return;
  const data=await response.json(),meta=renderer.meta;
  if(data.video_sha256!==meta.video_sha256||data.fps!==meta.fps||JSON.stringify(data.image_size)!==JSON.stringify(meta.image_size)||data.frames.length!==meta.frames)throw Error('球拍数据与当前视频不匹配');
  this.rows=data.frames;this.summary=data.summary;this.model=data.model;this.fps=data.fps;this.lastFrame=-1;
  this.transforms=this.rows.map(row=>{if(row.status!=='fitted')return null;const r=row.rotation_camera_columns,m=new THREE.Matrix4().set(...r[0],0,...r[1],0,...r[2],0,0,0,0,1),q=new THREE.Quaternion().setFromRotationMatrix(m);return {q,grip:new THREE.Vector3().fromArray(row.grip_target_camera_m||new THREE.Vector3().fromArray(row.translation_camera_m).add(new THREE.Vector3(0,this.model.grip_y_m,0).applyQuaternion(q)).toArray())}});this.tempQuaternion=new THREE.Quaternion();this.tempMatrix=new THREE.Matrix4();this.tempGrip=new THREE.Vector3();this.localGrip=new THREE.Vector3(0,this.model.grip_y_m,0);
  const responseMesh=await fetch('wilson_mesh.bin');if(!responseMesh.ok)throw Error('球拍模型加载失败');
  const values=new Float32Array(await responseMesh.arrayBuffer());if(!values.length||values.length%18||!values.every(Number.isFinite))throw Error('球拍模型无效');
  const buffer=new THREE.InterleavedBuffer(values,6),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));geometry.setAttribute('vertexColor',new THREE.InterleavedBufferAttribute(buffer,3,3));
  const uniforms={};for(const name of ['sourceRoot','basis','shift'])uniforms[name]=renderer.u[name];Object.assign(uniforms,{rotation:{value:new THREE.Matrix3()},translation:{value:new THREE.Vector3()},estimated:{value:0}});
  const material=new THREE.RawShaderMaterial({glslVersion:THREE.GLSL3,side:THREE.DoubleSide,uniforms,vertexShader:`precision highp float;in vec3 position;in vec3 vertexColor;uniform mat4 projectionMatrix,viewMatrix;uniform mat3 rotation,basis;uniform vec3 translation,sourceRoot,shift;out vec3 rgb;void main(){vec3 p=rotation*position+translation;gl_Position=projectionMatrix*viewMatrix*vec4(basis*(p-sourceRoot)+shift,1.);rgb=vertexColor;}`,fragmentShader:`precision highp float;in vec3 rgb;uniform float estimated;out vec4 color;void main(){color=vec4(mix(rgb,vec3(1.,.65,.22),estimated*.25),1.);}`});
  const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;mesh.visible=false;mesh.renderOrder=1;renderer.scene.add(mesh);this.state={mesh,uniforms};renderer.addLayer(this);
  const animate=()=>{if(!renderer.ready)return;if(!renderer.video.paused&&!renderer.video.seeking&&this.enabled&&this.lastFrame>=0){this.present(renderer,this.lastFrame);renderer.renderer.setRenderTarget(null);renderer.renderer.clear();renderer.renderer.render(renderer.scene,renderer.camera);}this.animation=requestAnimationFrame(animate)};this.animation=requestAnimationFrame(animate);
 }
 update(renderer,n,mode){
  if(!this.state)return;const row=this.rows[n];const label=document.getElementById('gripStatus');if(label)label.textContent=row?.status==='fitted'&&row.face_angle_to_camera_deg!==undefined?`掌内握点 · 拍面法线与相机 ${row.face_angle_to_camera_deg.toFixed(0)}° · ${row.quality==='silhouette_fitted'?'轮廓支持':'约束估计'}${row.mirror_observation_used?' · 镜中拍框支持':''}`:'';this.state.mesh.visible=this.enabled&&row?.status==='fitted';if(!this.state.mesh.visible)return;
  this.lastFrame=n;this.present(renderer,n);
 }
 present(renderer,n){
  const current=this.transforms[n],next=this.transforms[n+1];if(!current)return;const alpha=renderer.video.paused||renderer.video.seeking||!next?0:Math.max(0,Math.min(1,renderer.video.currentTime*this.fps-n));
  this.tempQuaternion.copy(current.q);if(alpha)this.tempQuaternion.slerp(next.q,alpha);this.tempMatrix.makeRotationFromQuaternion(this.tempQuaternion);this.state.uniforms.rotation.value.setFromMatrix4(this.tempMatrix);
  this.tempGrip.copy(current.grip);if(alpha)this.tempGrip.lerp(next.grip,alpha);this.state.uniforms.translation.value.copy(this.tempGrip).sub(this.localGrip.clone().applyQuaternion(this.tempQuaternion));
  // Blend evidence tint between frames instead of flashing green/orange at the threshold.
  const a=this.rows[n].quality==='silhouette_fitted'?0:1,b=this.rows[n+1]?.quality==='silhouette_fitted'?0:1;this.state.uniforms.estimated.value=a*(1-alpha)+b*alpha;
 }
 dispose(){cancelAnimationFrame(this.animation);this.state?.mesh.geometry.dispose();this.state?.mesh.material.dispose();}
 overlay(context,transform,n){
  const row=this.rows[n];if(!this.enabled||row?.status!=='fitted')return;context.strokeStyle=row.quality!=='silhouette_fitted'?'#ffbd62':'#6ce5a3';context.lineWidth=1.5;context.beginPath();row.projected_head_outline?.forEach((p,i)=>{const q=transform(p);i?context.lineTo(...q):context.moveTo(...q)});context.closePath();context.stroke();
 }
}
