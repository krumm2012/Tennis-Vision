// Current-video Wilson poses share the SAM display transform and camera uniforms.
class DatasetRacketLayer {
 constructor(){this.enabled=true;this.rows=[];this.state=null;}
 async load(renderer){
  const response=await fetch('racket_poses.json',{cache:'no-store'});if(!response.ok)return;
  const data=await response.json(),meta=renderer.meta;
  if(data.video_sha256!==meta.video_sha256||data.fps!==meta.fps||JSON.stringify(data.image_size)!==JSON.stringify(meta.image_size)||data.frames.length!==meta.frames)throw Error('球拍数据与当前视频不匹配');
  this.rows=data.frames;this.summary=data.summary;this.model=data.model;
  const responseMesh=await fetch('wilson_mesh.bin');if(!responseMesh.ok)throw Error('球拍模型加载失败');
  const values=new Float32Array(await responseMesh.arrayBuffer());if(!values.length||values.length%18||!values.every(Number.isFinite))throw Error('球拍模型无效');
  const buffer=new THREE.InterleavedBuffer(values,6),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));geometry.setAttribute('vertexColor',new THREE.InterleavedBufferAttribute(buffer,3,3));
  const uniforms={};for(const name of ['sourceRoot','basis','shift'])uniforms[name]=renderer.u[name];Object.assign(uniforms,{rotation:{value:new THREE.Matrix3()},translation:{value:new THREE.Vector3()},estimated:{value:0}});
  const material=new THREE.RawShaderMaterial({glslVersion:THREE.GLSL3,side:THREE.DoubleSide,uniforms,vertexShader:`precision highp float;in vec3 position;in vec3 vertexColor;uniform mat4 projectionMatrix,viewMatrix;uniform mat3 rotation,basis;uniform vec3 translation,sourceRoot,shift;out vec3 rgb;void main(){vec3 p=rotation*position+translation;gl_Position=projectionMatrix*viewMatrix*vec4(basis*(p-sourceRoot)+shift,1.);rgb=vertexColor;}`,fragmentShader:`precision highp float;in vec3 rgb;uniform float estimated;out vec4 color;void main(){color=vec4(mix(rgb,vec3(1.,.65,.22),estimated*.5),1.);}`});
  const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;mesh.visible=false;mesh.renderOrder=1;renderer.scene.add(mesh);this.state={mesh,uniforms};renderer.addLayer(this);
 }
 update(renderer,n,mode){
  if(!this.state)return;const row=this.rows[n];this.state.mesh.visible=this.enabled&&row?.status==='fitted';if(!this.state.mesh.visible)return;
  this.state.uniforms.rotation.value.set(...row.rotation_camera_columns.flat());this.state.uniforms.translation.value.fromArray(row.translation_camera_m);this.state.uniforms.estimated.value=row.quality==='interpolated'?1:0;
 }
 overlay(context,transform,n){
  const row=this.rows[n];if(!this.enabled||row?.status!=='fitted')return;context.strokeStyle=row.quality==='interpolated'?'#ffbd62':'#6ce5a3';context.lineWidth=1.5;context.beginPath();row.projected_head_outline?.forEach((p,i)=>{const q=transform(p);i?context.lineTo(...q):context.moveTo(...q)});context.closePath();context.stroke();
 }
}
