// Optional UV appearance on the exact displayed body, including seam duplicates.
class DatasetAppearanceLayer {
 constructor(){this.enabled=true;}
 async load(renderer){
  const response=await fetch('appearance_manifest.json',{cache:'no-store'});if(response.status===404)return false;if(!response.ok)throw Error('合成纹理清单加载失败');
  const manifest=await response.json();
  if(manifest.video_sha256!==renderer.meta.video_sha256||manifest.vertices!==renderer.meta.vertices)throw Error('合成纹理与当前人体不匹配');
  const read=async name=>{const r=await fetch(name,{cache:'no-store'});if(!r.ok)throw Error('合成纹理文件缺失');const data=await r.arrayBuffer();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),v=>v.toString(16).padStart(2,'0')).join('');if(hash!==manifest.files[name])throw Error('合成纹理文件版本不一致');return data;};
  const [mapping,uv,indices,png]=await Promise.all(['appearance_map.bin','appearance_uv.bin','appearance_indices.bin','body_texture_rgba.png'].map(read));
  this.mapping=new Uint32Array(mapping);const coords=new Float32Array(uv),faces=new Uint32Array(indices);
  if(coords.length!==this.mapping.length*2||!coords.every(Number.isFinite)||this.mapping.some(v=>v>=renderer.meta.vertices)||faces.length!==renderer.faces.length||faces.some((v,i)=>v>=this.mapping.length||this.mapping[v]!==renderer.faces[i]))throw Error('合成纹理拓扑不匹配');
  const url=URL.createObjectURL(new Blob([png],{type:'image/png'})),image=new Image();try{image.src=url;await image.decode();}finally{URL.revokeObjectURL(url);}
  this.texture=new THREE.Texture(image);this.texture.flipY=true;this.texture.colorSpace=THREE.NoColorSpace;this.texture.needsUpdate=true;
  this.geometry=new THREE.BufferGeometry();this.geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(this.mapping.length*3),3).setUsage(THREE.DynamicDrawUsage));this.geometry.setAttribute('uv',new THREE.BufferAttribute(coords,2));this.geometry.setIndex(new THREE.BufferAttribute(faces,1));
  this.material=new THREE.RawShaderMaterial({glslVersion:THREE.GLSL3,side:THREE.DoubleSide,uniforms:{basis:renderer.u.basis,shift:renderer.u.shift,atlas:{value:this.texture}},vertexShader:`precision highp float;in vec3 position;in vec2 uv;uniform mat4 projectionMatrix,viewMatrix;uniform mat3 basis;uniform vec3 shift;out vec2 coords;void main(){coords=uv;gl_Position=projectionMatrix*viewMatrix*vec4(basis*position+shift,1.);}`,fragmentShader:`precision highp float;in vec2 coords;uniform sampler2D atlas;out vec4 color;void main(){vec4 t=texture(atlas,coords);color=vec4(mix(vec3(.32,.40,.46),t.rgb,t.a),1.);}`});
  this.mesh=new THREE.Mesh(this.geometry,this.material);this.mesh.frustumCulled=false;this.mesh.visible=false;renderer.scene.add(this.mesh);renderer.addLayer(this);this.manifest=manifest;return true;
 }
 update(renderer,n,mode){
  if(!this.mesh)return;this.mesh.visible=this.enabled&&renderer.body.visible;
  if(!this.mesh.visible)return;
  // Source/depth/diagnostic passes continue using the original unexpanded topology.
  renderer.body.visible=false;
  if(this.frame===n&&this.mode===mode)return;
  const source=renderer.geometryBuffer.getAttribute('displayPosition').array,target=this.geometry.getAttribute('position');
  for(let i=0;i<this.mapping.length;i++){const v=this.mapping[i]*3;target.array[i*3]=source[v];target.array[i*3+1]=source[v+1];target.array[i*3+2]=source[v+2];}
  target.needsUpdate=true;this.frame=n;this.mode=mode;
 }
 dispose(){this.mesh?.removeFromParent();this.geometry?.dispose();this.material?.dispose();this.texture?.dispose();}
}
