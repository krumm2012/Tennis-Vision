// CPU integration regression: the racket must share the body decoded frame and root.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'../..'),scene=vm.createContext({console,AbortController,TextDecoder,TextEncoder,performance,requestAnimationFrame:()=>0,cancelAnimationFrame:()=>{},document:{getElementById:()=>null}});
vm.runInContext(fs.readFileSync(path.join(root,'viewer/sam3d/vendor/three-0.180.0.min.js'),'utf8'),scene);
vm.runInContext(fs.readFileSync(process.argv[2]||path.join(__dirname,'dataset_racket.js'),'utf8'),scene);
const THREE=scene.THREE;
const roots=[[0,0,4],[10,20,8]],rows=roots.map((r,frame)=>({frame,status:'fitted',quality:'silhouette_fitted',rotation_camera_columns:[[1,0,0],[0,1,0],[0,0,1]],translation_camera_m:[r[0]+1,r[1]+2-.045,r[2]],grip_target_camera_m:[r[0]+1,r[1]+2,r[2]]}));
const meta={video_sha256:'this-video',fps:25,image_size:[2560,1440],frames:2,source_roots:roots};
scene.fetch=async url=>({ok:true,json:async()=>({...meta,model:{grip_y_m:.045},frames:rows}),arrayBuffer:async()=>new Float32Array(18).fill(.1).buffer});
const renderer={meta,scene:new THREE.Scene(),u:{sourceRoot:{value:new THREE.Vector3().fromArray(roots[0])},basis:{value:new THREE.Matrix3()},shift:{value:new THREE.Vector3()}},video:{paused:false,seeking:false,currentTime:.02},addLayer:()=>{}};
(async()=>{
 const layer=vm.runInContext('new DatasetRacketLayer()',scene);await layer.load(renderer);
 layer.present(renderer,0);
 const grip=layer.state.uniforms.translation.value.clone().add(new THREE.Vector3(0,.045,0)).sub(renderer.u.sourceRoot.value);
 assert.ok(grip.distanceTo(new THREE.Vector3(1,2,0))<1e-10,'interpolating camera roots caused spurious grip displacement');
 renderer.video.paused=true;layer.present(renderer,0);assert.ok(layer.state.uniforms.translation.value.distanceTo(new THREE.Vector3(1,1.955,4))<1e-10);
 renderer.video.paused=false;renderer.video.currentTime=.039;
 layer.transforms[1].grip.x+=.3;layer.present(renderer,0);
 const lockedGrip=layer.state.uniforms.translation.value.clone().add(new THREE.Vector3(0,.045,0)).sub(renderer.u.sourceRoot.value);
 assert.ok(lockedGrip.distanceTo(new THREE.Vector3(1,2,0))<1e-10,'racket advances toward next frame while body remains at decoded frame');
 console.log('PASS: source-root invariance and body/racket decoded-frame synchronization');
})().catch(error=>{console.error(error);process.exitCode=1});
