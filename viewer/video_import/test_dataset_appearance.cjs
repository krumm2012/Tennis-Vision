// Exercise actual UV layer with seam duplication and pose switching without WebGL.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const context=vm.createContext({console,AbortController,TextDecoder,TextEncoder,performance,crypto:crypto.webcrypto,Blob,URL,Image:class{async decode(){}},fetch:async name=>({status:200,ok:true,json:async()=>manifest,arrayBuffer:async()=>files[name].buffer})});
vm.runInContext(fs.readFileSync('viewer/sam3d/vendor/three-0.180.0.min.js','utf8'),context);
vm.runInContext(fs.readFileSync('viewer/video_import/dataset_appearance.js','utf8'),context);
const T=context.THREE,files={'appearance_map.bin':new Uint32Array([0,1,2,0]),'appearance_uv.bin':new Float32Array(8),'appearance_indices.bin':new Uint32Array([3,1,2]),'body_texture_rgba.png':new Uint8Array([1,2,3])};
const manifest={video_sha256:'a',vertices:3,files:Object.fromEntries(Object.entries(files).map(([name,value])=>[name,crypto.createHash('sha256').update(new Uint8Array(value.buffer)).digest('hex')]))};
const renderer={meta:{video_sha256:'a',vertices:3},faces:new Uint32Array([0,1,2]),u:{basis:{value:new T.Matrix3()},shift:{value:new T.Vector3()}},scene:new T.Scene(),body:{visible:true},geometryBuffer:new T.BufferGeometry(),addLayer(){}};
renderer.geometryBuffer.setAttribute('displayPosition',new T.BufferAttribute(new Float32Array([1,2,3,4,5,6,7,8,9]),3));
(async()=>{
 const layer=vm.runInContext('new DatasetAppearanceLayer()',context);assert.equal(await layer.load(renderer),true);layer.update(renderer,0,'raw');assert.equal(renderer.body.visible,false);assert.equal(layer.mesh.visible,true);
 assert.deepEqual(Array.from(layer.geometry.attributes.position.array),[1,2,3,4,5,6,7,8,9,1,2,3]);
 renderer.body.visible=true;renderer.geometryBuffer.attributes.displayPosition.array[0]=10;layer.update(renderer,0,'refined');assert.equal(layer.geometry.attributes.position.array[9],10,'UV seam must follow displayed pose');
 renderer.body.visible=true;layer.enabled=false;layer.update(renderer,0,'refined');assert.equal(layer.mesh.visible,false);assert.equal(renderer.body.visible,true,'projection fallback must restore body');
 files['appearance_map.bin'][0]=2;await assert.rejects(vm.runInContext('new DatasetAppearanceLayer()',context).load(renderer),/版本不一致/);
 layer.dispose();console.log('PASS: atlas provenance, UV seams, displayed pose and projection fallback');
})().catch(e=>{console.error(e);process.exitCode=1});
