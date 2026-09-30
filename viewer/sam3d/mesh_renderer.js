// Three.js owns all GPU resources; source-camera projection stays in custom GLSL.
class SamMeshRenderer {
 static textureLayout(meta,image){
  // Legacy assets were exported in 1280x720 camera coordinates and a 16x16 atlas.
  const size=meta.image_size||[1280,720],grid=meta.mask_atlas_grid||[16,16];
  if([...size,...grid].some(x=>!Number.isInteger(x)||x<=0)||size.length!==2||grid.length!==2||grid[0]*grid[1]<meta.frames||image.width%grid[0]||image.height%grid[1])throw Error('贴图尺寸或遮罩图集布局无效');
  return {size,grid,tile:[image.width/grid[0],image.height/grid[1]]};
 }
 constructor(canvas,video){
  this.canvas=canvas;this.video=video;this.ready=false;this.meshFrames=new Map();
  this.meshPending=new Map();this.meshAbort=new Map();this.meshWhole=new Map();this.layers=[];
  this.renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true,premultipliedAlpha:false,preserveDrawingBuffer:true});
  this.renderer.autoClear=false;this.renderer.outputColorSpace=THREE.SRGBColorSpace;
  this.renderer.setClearColor(0,0);this.scene=new THREE.Scene();this.bodyScene=new THREE.Scene();
  this.camera=new THREE.OrthographicCamera(-1,1,1,-1,0,40);
  this.camera.matrixAutoUpdate=false;this.camera.matrixWorldAutoUpdate=false;
  this.guides=new THREE.Group();this.guideLines=new Map();
  this.jointDots=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({vertexColors:true,size:6,depthTest:false}));
  this.guides.frustumCulled=this.jointDots.frustumCulled=false;this.jointDots.renderOrder=3;
  this.scene.add(this.guides,this.jointDots);
 }
 addLayer(layer){this.layers.push(layer);}
 async frameMesh(file,n){const key=file+':'+n;if(this.meshFrames.has(key)){const value=this.meshFrames.get(key);this.meshFrames.delete(key);this.meshFrames.set(key,value);return value}const count=this.meta.vertices*3;if(this.meshWhole.has(file))return this.meshWhole.get(file).subarray(n*count,(n+1)*count);if(this.meshPending.has(key))return this.meshPending.get(key);const begin=n*count*4,end=begin+count*4-1,controller=new AbortController();this.meshAbort.set(key,controller);const request=fetch(file,{headers:{Range:`bytes=${begin}-${end}`},signal:controller.signal}).then(async r=>{if(!r.ok)throw Error(file+' 下载失败（HTTP '+r.status+'）');const bytes=await r.arrayBuffer();let value;if(r.status===206){if(bytes.byteLength!==count*4)throw Error(file+' 帧数据长度错误');value=new Float32Array(bytes)}else if(r.status===200&&bytes.byteLength% (count*4)===0){const all=new Float32Array(bytes);this.meshWhole.set(file,all);value=all.subarray(n*count,(n+1)*count)}else throw Error(file+' 无法读取网格帧（HTTP '+r.status+'）');this.meshFrames.set(key,value);while(this.meshFrames.size>40)this.meshFrames.delete(this.meshFrames.keys().next().value);this.onFrameReady?.(n);return value}).catch(e=>{if(e.name!=='AbortError')this.onError?.(e,n);throw e}).finally(()=>{this.meshPending.delete(key);this.meshAbort.delete(key)});this.meshPending.set(key,request);return request}
 prefetch(n,files){if(!this.meshAbort)this.meshAbort=new Map();for(const [key,controller] of this.meshAbort){const frame=Number(key.slice(key.lastIndexOf(':')+1));if(frame<n-2||frame>n+12)controller.abort()}const end=Math.min(this.meta.frames-1,n+8);for(const file of new Set(files))for(let frame=n;frame<=end;frame++)this.frameMesh(file,frame).catch(()=>{});}
 async load(){const checked=async url=>{const r=await fetch(url);if(!r.ok)throw Error(url+' 加载失败（HTTP '+r.status+'）');return r;};this.temporalTexture=new Uint8Array(await checked('temporal_texture_sam2.bin').then(r=>r.arrayBuffer()));this.meta=await checked('mesh_meta.json').then(r=>r.json());this.mirror=await checked('mirror_geometry_frames.json').then(r=>r.json());this.geometry=await checked('mirror_geometry.json').then(r=>r.json());this.maskImage=new Image();this.maskImage.src='person_masks_sam2.png';await this.maskImage.decode();this.maskStats=await checked('person_masks_sam2_stats.json').then(r=>{if(!r.ok)throw Error('遮罩统计加载失败');return r.json()});this.faces=new Uint32Array(await checked('mesh_faces.bin').then(r=>{if(!r.ok)throw Error('网格三角面下载失败');return r.arrayBuffer()}));
 this.layout=SamMeshRenderer.textureLayout(this.meta,this.maskImage);
 const vs=` precision highp float; precision highp int;
 in vec4 cachedColor;in vec2 cachedMeta;uniform bool mirrorCacheAllowed;out vec4 priorColor;in vec3 rawPosition; in vec3 displayPosition;
 uniform vec2 sourceSize;uniform mat4 projectionMatrix;uniform mat4 viewMatrix;uniform vec3 sourceRoot; uniform float focal; uniform mat3 basis; uniform vec3 shift;uniform vec3 center;uniform vec2 viewport;uniform vec2 angles;uniform float scale;uniform int sourcePass;uniform vec4 mirrorU;uniform vec4 mirrorV;uniform vec3 mirrorDir;uniform vec4 mirrorPlane;
 out vec3 originalCamera;
 vec4 sourceClip(vec3 p){float n=.1,f=100.;return vec4(2.*focal*p.x/sourceSize.x,-2.*focal*p.y/sourceSize.y,(f+n)/(f-n)*p.z-2.*f*n/(f-n),p.z);}
 void main(){float ca=cachedColor.a;if(cachedMeta.x>1.5&&!mirrorCacheAllowed)ca=0.;priorColor=vec4(cachedColor.rgb*ca,ca);originalCamera=rawPosition+sourceRoot;if(sourcePass==1){gl_Position=sourceClip(originalCamera);return;}if(sourcePass==2){vec3 reflected=originalCamera-2.*(dot(originalCamera,mirrorPlane.xyz)-mirrorPlane.w)*mirrorPlane.xyz;gl_Position=sourceClip(reflected);return;}gl_Position=projectionMatrix*viewMatrix*vec4(basis*displayPosition+shift,1.);}`;
 const fs=`precision highp float;precision highp int;
 in vec4 priorColor;in vec3 originalCamera;out vec4 color;
 uniform bool temporalEnabled,statsPass,textured,mirrorEnabled,maskEnabled,diagnostic,edgeRepair;
 uniform int sourcePass;uniform float focal;
 uniform sampler2D frameTexture,sourceDepth,mirrorDepth,personMasks;
 uniform vec4 mirrorPlane;uniform vec2 maskTile,atlasGrid,maskSize,sourceSize;
 vec2 maskUV(vec2 uv){return (vec2(maskTile.x,atlasGrid.y-1.-maskTile.y)+clamp(uv,.5/maskSize,1.-.5/maskSize))/atlasGrid;}
 bool inImage(vec2 uv){return all(greaterThanEqual(uv,vec2(0.)))&&all(lessThanEqual(uv,vec2(1.)));}
 float maskConfidence(vec2 uv,bool mirror){vec2 m=texture(personMasks,maskUV(uv)).rg;return mirror?m.g:m.r;}
 float viewDepth(vec2 uv,bool mirror){return mirror?texture(mirrorDepth,uv).r:texture(sourceDepth,uv).r;}
 vec2 project(vec3 p){return vec2(.5+focal*p.x/(sourceSize.x*p.z),.5-focal*p.y/(sourceSize.y*p.z));}
 float projectedDepth(float z){float n=.1,f=100.;return .5*((f+n)/(f-n)-2.*f*n/((f-n)*z))+.5;}
 // Keep the projected surface's visibility test. Edge repair cannot reveal hidden limbs.
 vec4 observation(vec3 p,bool mirror){
  if(p.z<=.1)return vec4(0.);
  vec2 uv=project(p);if(!inImage(uv))return vec4(0.);
  float depth=projectedDepth(p.z),seen=viewDepth(uv,mirror);
  if(depth>seen+.000018||(mirror&&texture(sourceDepth,uv).r<.9999))return vec4(0.);
  float confidence=maskConfidence(uv,mirror);
  float weight=maskEnabled?smoothstep(.55,.85,confidence):1.;vec2 chosen=uv;
  // One mask texel only. Move the COLOR lookup into the mask, never merely dilate it.
  // Recovered observations remain lower confidence than direct interior pixels.
  if(maskEnabled&&edgeRepair&&weight<.7){
   for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
    if(x==0&&y==0)continue;
    vec2 offset=vec2(float(x),float(y)),candidate=uv+offset/maskSize;
    if(!inImage(candidate))continue;
    float m=maskConfidence(candidate,mirror);
    if(m<.85||abs(viewDepth(candidate,mirror)-depth)>.000018)continue;
    if(mirror&&texture(sourceDepth,candidate).r<.9999)continue;
    float score=.7*(1.-.15*length(offset))*smoothstep(.85,.98,m);
    if(score>weight){weight=score;chosen=candidate;}
   }
  }
  return vec4(texture(frameTexture,chosen).rgb,weight);
 }
 void main(){
  if(sourcePass!=0){color=vec4(0.);return;}
  vec3 p=originalCamera;vec4 direct=observation(p,false),reflected=vec4(0.);
  if(mirrorEnabled&&direct.a<1.)reflected=observation(p-2.*(dot(p,mirrorPlane.xyz)-mirrorPlane.w)*mirrorPlane.xyz,true);
  vec3 normal=normalize(cross(dFdx(p),dFdy(p)));float light=.48+.52*abs(dot(normal,normalize(vec3(-.4,-.6,-1.))));vec3 neutral=vec3(.40,.48,.56)*light;
  float dw=textured?direct.a:0.,mw=textured?reflected.a*(1.-dw):0.;
  float cw=(textured&&temporalEnabled&&maskEnabled)?min(.85,priorColor.a)*(1.-dw-mw):0.;float remaining=max(0.,1.-dw-mw-cw);
  if(diagnostic){color=statsPass?vec4(dw,mw,remaining,cw):vec4(dw+cw,mw+cw,remaining,1.);return;}
  vec3 cached=priorColor.rgb/max(priorColor.a,.0001);color=vec4(direct.rgb*dw+reflected.rgb*mw+cached*cw+neutral*remaining,1.);
 }`;
 const uniform=value=>({value});
 this.u={
  sourceSize:uniform(new THREE.Vector2(...this.layout.size)),atlasGrid:uniform(new THREE.Vector2(...this.layout.grid)),maskSize:uniform(new THREE.Vector2(...this.layout.tile)),edgeRepair:uniform(true),
  sourceRoot:uniform(new THREE.Vector3()),focal:uniform(1),basis:uniform(new THREE.Matrix3()),
  shift:uniform(new THREE.Vector3()),center:uniform(new THREE.Vector3()),viewport:uniform(new THREE.Vector2()),
  angles:uniform(new THREE.Vector2()),scale:uniform(1),sourcePass:uniform(0),
  mirrorPlane:uniform(new THREE.Vector4()),mirrorU:uniform(new THREE.Vector4()),mirrorV:uniform(new THREE.Vector4()),mirrorDir:uniform(new THREE.Vector3()),
  textured:uniform(true),mirrorEnabled:uniform(false),mirrorCacheAllowed:uniform(false),
  temporalEnabled:uniform(false),maskEnabled:uniform(true),diagnostic:uniform(false),statsPass:uniform(false),maskTile:uniform(new THREE.Vector2())
 };
 this.geometryBuffer=new THREE.BufferGeometry();
 for(const name of ['rawPosition','displayPosition'])this.geometryBuffer.setAttribute(name,new THREE.BufferAttribute(new Float32Array(this.meta.vertices*3),3).setUsage(THREE.DynamicDrawUsage));
 this.cacheArray=new Uint8Array(this.meta.vertices*6);
 this.cacheBuffer=new THREE.InterleavedBuffer(this.cacheArray,6).setUsage(THREE.DynamicDrawUsage);
 this.geometryBuffer.setAttribute('cachedColor',new THREE.InterleavedBufferAttribute(this.cacheBuffer,4,0,true));
 this.geometryBuffer.setAttribute('cachedMeta',new THREE.InterleavedBufferAttribute(this.cacheBuffer,2,4,false));
 this.geometryBuffer.setIndex(new THREE.BufferAttribute(this.faces,1));
 this.texture=new THREE.VideoTexture(this.video);this.texture.flipY=true;this.texture.colorSpace=THREE.NoColorSpace;
 this.maskTexture=new THREE.Texture(this.maskImage);this.maskTexture.flipY=true;this.maskTexture.colorSpace=THREE.NoColorSpace;
 this.maskTexture.minFilter=this.maskTexture.magFilter=THREE.LinearFilter;this.maskTexture.generateMipmaps=false;this.maskTexture.needsUpdate=true;
 const depthTarget=()=>{const target=new THREE.WebGLRenderTarget(...this.layout.size,{depthBuffer:true});target.depthTexture=new THREE.DepthTexture(...this.layout.size,THREE.UnsignedIntType);return target;};
 this.sourceTarget=depthTarget();this.mirrorTarget=depthTarget();
 this.statsTarget=new THREE.WebGLRenderTarget(160,120,{depthBuffer:true});
 Object.assign(this.u,{frameTexture:uniform(this.texture),personMasks:uniform(this.maskTexture),sourceDepth:uniform(null),mirrorDepth:uniform(null)});
 this.material=new THREE.RawShaderMaterial({vertexShader:vs,fragmentShader:fs,uniforms:this.u,glslVersion:THREE.GLSL3,side:THREE.DoubleSide,blending:THREE.NoBlending});
 this.body=new THREE.Mesh(this.geometryBuffer,this.material);this.body.frustumCulled=false;this.body.renderOrder=0;this.scene.add(this.body);
 this.passBody=new THREE.Mesh(this.geometryBuffer,this.material);this.passBody.frustumCulled=false;this.bodyScene.add(this.passBody);
 this.statsPixels=new Uint8Array(160*120*4);this.ready=true;
 }
 async ensureFrame(n,mode){
  const file=this.displayFile(mode);await Promise.all([...new Set(['mesh_local.bin',file])].map(file=>this.frameMesh(file,n)));
 }
 displayFile(mode){return mode==='temporal'?'mesh_temporal.bin':mode==='refined'?'mesh_refined.bin':mode==='raw'?'mesh_local.bin':'mesh_smooth.bin';}
 setGuides(segments,points){
  const key=JSON.stringify([segments,points]);if(key===this.guidesKey)return;this.guidesKey=key;
  const groups=new Map();
  for(const row of segments){const key=JSON.stringify([row.width||1,row.dashed||false]);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
  for(const [key,line] of this.guideLines)line.visible=groups.has(key);
  const arrays=rows=>{const positions=[],colors=[];for(const row of rows){const color=new THREE.Color(row.color);for(const p of row.points){positions.push(...p);colors.push(color.r,color.g,color.b);}}return{positions,colors};};
  for(const [key,rows] of groups){
   let line=this.guideLines.get(key);const [width,dashed]=JSON.parse(key);
   if(!line){line=new THREE.LineSegments2(new THREE.LineSegmentsGeometry(),new THREE.LineMaterial({vertexColors:true,linewidth:width,dashed,dashSize:.025,gapSize:.025}));line.frustumCulled=false;this.guideLines.set(key,line);this.guides.add(line);}
   const {positions,colors}=arrays(rows),count=positions.length/6;
   if(!line.capacity||count>line.capacity){line.capacity=Math.max(count,(line.capacity||8)*2);line.geometry.dispose();line.geometry=new THREE.LineSegmentsGeometry();line.geometry.setPositions(new Float32Array(line.capacity*6));line.geometry.setColors(new Float32Array(line.capacity*6));}
   const attrs=line.geometry.attributes;attrs.instanceStart.data.array.set(positions);attrs.instanceStart.data.needsUpdate=true;attrs.instanceColorStart.data.array.set(colors);attrs.instanceColorStart.data.needsUpdate=true;line.geometry.instanceCount=count;
   if(dashed)line.computeLineDistances();line.visible=true;
  }
  const {positions,colors}=arrays(points),count=positions.length/3;
  if(!this.pointCapacity||count>this.pointCapacity){this.pointCapacity=Math.max(count,(this.pointCapacity||16)*2);this.jointDots.geometry.dispose();this.jointDots.geometry=new THREE.BufferGeometry();for(const name of ['position','color'])this.jointDots.geometry.setAttribute(name,new THREE.Float32BufferAttribute(new Float32Array(this.pointCapacity*3),3));}
  const attrs=this.jointDots.geometry.attributes;attrs.position.array.set(positions);attrs.position.needsUpdate=true;attrs.color.array.set(colors);attrs.color.needsUpdate=true;this.jointDots.geometry.setDrawRange(0,count);
 }
 setView(center,angles,scale,rect){
  const [yaw,pitch]=angles,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const x=[cy,0,sy],y=[sp*sy,cp,-sp*cy],z=[-cp*sy,sp,cp*cy],dot=a=>a.reduce((s,v,i)=>s+v*center[i],0);
  this.camera.matrixWorldInverse.set(...x,-dot(x),...y,-dot(y),...z,-dot(z)-20,0,0,0,1);
  this.camera.matrixWorld.copy(this.camera.matrixWorldInverse).invert();
  Object.assign(this.camera,{left:-rect.width/(2*scale),right:rect.width/(2*scale),top:rect.height/(2*scale),bottom:-rect.height/(2*scale)});this.camera.updateProjectionMatrix();
 }
 draw(n,mode,basis,shift,center,angles,scale,rect,textured=true,useMirror=false,useMask=true,bodyVisible=true){
  if(!this.ready||this.video.readyState<2||this.video.seeking)return false;
  const file=this.displayFile(mode),count=this.meta.vertices*3;
  if(bodyVisible)this.prefetch(n,['mesh_local.bin',file]);
  const frame=file=>this.meshFrames.get(file+':'+n)||this.meshWhole.get(file)?.subarray(n*count,(n+1)*count);
  const raw=frame('mesh_local.bin'),display=frame(file);if(bodyVisible&&(!raw||!display))return false;
  const renderer=this.renderer,u=this.u,d=devicePixelRatio||1;
  // Upload even for an already-decoded first frame or a paused seek.
  this.texture.needsUpdate=true;
  if(this.canvas.width!==Math.round(rect.width*d)||this.canvas.height!==Math.round(rect.height*d)){
   renderer.setPixelRatio(d);renderer.setSize(rect.width,rect.height,false);
  }
  if(bodyVisible&&this.uploadedFrame!==n){const attr=this.geometryBuffer.getAttribute('rawPosition');attr.array.set(raw);attr.needsUpdate=true;this.cacheArray.set(this.temporalTexture.subarray(n*this.meta.vertices*6,(n+1)*this.meta.vertices*6));this.cacheBuffer.needsUpdate=true;}
  if(bodyVisible&&(this.uploadedFrame!==n||this.uploadedMode!==mode)){const attr=this.geometryBuffer.getAttribute('displayPosition');attr.array.set(display);attr.needsUpdate=true;}
  if(bodyVisible){this.uploadedFrame=n;this.uploadedMode=mode;}
  this.body.visible=bodyVisible;this.setView(center,angles,scale,rect);
  for(const line of this.guideLines.values())line.material.resolution.set(rect.width,rect.height);
  u.sourceRoot.value.fromArray(this.meta.source_roots[n]);u.focal.value=this.meta.focal[n];u.basis.value.fromArray(basis.flat());
  u.shift.value.fromArray(shift);u.center.value.fromArray(center);u.viewport.value.set(rect.width,rect.height);u.angles.value.fromArray(angles);u.scale.value=scale;
  const enabled=!!(useMirror&&this.maskStats[n].mirror>0);
  u.temporalEnabled.value=!!this.useTemporalTexture;u.textured.value=textured;u.mirrorEnabled.value=enabled;u.mirrorCacheAllowed.value=useMirror;
  u.maskEnabled.value=useMask;u.edgeRepair.value=this.edgeRepair!==false;u.maskTile.value.set(n%this.layout.grid[0],Math.floor(n/this.layout.grid[0]));u.mirrorPlane.value.fromArray([...this.geometry.normal_camera,this.geometry.distance_camera_m]);
  u.diagnostic.value=!!this.showSources;u.statsPass.value=false;
  if(bodyVisible){
  // Never sample the texture attached to the current depth target.
  u.sourceDepth.value=null;u.mirrorDepth.value=null;u.sourcePass.value=1;
  renderer.setRenderTarget(this.sourceTarget);renderer.clear();renderer.render(this.bodyScene,this.camera);
  if(enabled){u.sourcePass.value=2;renderer.setRenderTarget(this.mirrorTarget);renderer.clear();renderer.render(this.bodyScene,this.camera);}
  u.sourcePass.value=0;u.sourceDepth.value=this.sourceTarget.depthTexture;u.mirrorDepth.value=this.mirrorTarget.depthTexture;
  }
  for(const layer of this.layers)layer.update(this,n,mode);
  renderer.setRenderTarget(null);renderer.clear();renderer.render(this.scene,this.camera);
  if(this.collectStats&&bodyVisible&&(this.video.paused||!this.statsTime||performance.now()-this.statsTime>250)){
   u.statsPass.value=true;u.diagnostic.value=true;
   renderer.setRenderTarget(this.statsTarget);renderer.clear();renderer.render(this.bodyScene,this.camera);
   renderer.readRenderTargetPixels(this.statsTarget,0,0,160,120,this.statsPixels);
   let total=0,real=0,mirror=0,gray=0,temporal=0;
   for(let k=0;k<this.statsPixels.length;k+=4){if(this.statsPixels[k]+this.statsPixels[k+1]+this.statsPixels[k+2]+this.statsPixels[k+3]>0){total++;real+=this.statsPixels[k]/255;mirror+=this.statsPixels[k+1]/255;gray+=this.statsPixels[k+2]/255;temporal+=this.statsPixels[k+3]/255;}}
   this.coverage={frame:n,total,real:total?real/total:0,mirror:total?mirror/total:0,gray:total?gray/total:0,temporal:total?temporal/total:0};
   this.statsTime=performance.now();renderer.setRenderTarget(null);u.statsPass.value=false;u.diagnostic.value=!!this.showSources;
  }
  return true;
 }
 dispose(){
  for(const controller of this.meshAbort.values())controller.abort();
  for(const layer of this.layers)layer.dispose?.();
  for(const resource of [this.geometryBuffer,this.material,this.texture,this.maskTexture,this.sourceTarget,this.mirrorTarget,this.statsTarget])resource?.dispose();
  for(const object of [...this.guideLines.values(),this.jointDots]){object.geometry.dispose();object.material.dispose();}
  this.renderer.dispose();this.ready=false;
 }
}
