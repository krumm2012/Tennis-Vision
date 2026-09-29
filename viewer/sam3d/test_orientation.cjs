const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context={console,AbortController};vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/vendor/three-0.180.0.min.js','utf8'),context);
const source=fs.readFileSync(__dirname+'/mesh_renderer.js','utf8');
vm.runInContext(source+'\nthis.Renderer=SamMeshRenderer;',context);
const T=context.THREE,renderer=Object.create(context.Renderer.prototype);
renderer.camera=new T.OrthographicCamera(-1,1,1,-1,0,40);
// Compare the actual Three.js camera matrices with the established viewer projection.
let cases=0;
for(const center of [[0,0,0],[1.65,.6,2.4]])for(const yaw of [0,.3,Math.PI,-1])for(const pitch of [-.5,0,.2,.8])for(const point of [[1,2,3],[-1,0,1],[0,1,-1]]){
 renderer.setView(center,[yaw,pitch],100,{width:640,height:450});
 const p=point.map((v,i)=>v-center[i]),[x,y,z]=p,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const xx=x*cy+z*sy,zz=-x*sy+z*cy,yy=y*cp-zz*sp,depth=y*sp+zz*cp;
 const actual=new T.Vector3(...point).applyMatrix4(renderer.camera.matrixWorldInverse).applyMatrix4(renderer.camera.projectionMatrix);
 const expected=[xx*200/640,yy*200/450,-depth/20];
 expected.forEach((value,i)=>assert(Math.abs(actual.toArray()[i]-value)<1e-10,`camera mismatch ${i}`));cases++;
}
assert(source.includes('if(sourcePass==1){gl_Position=sourceClip(originalCamera);return;}'));
assert(source.includes('gl_Position=sourceClip(reflected);return;'));
assert(source.includes('(f+n)/(f-n)*p.z-2.*f*n/(f-n),p.z)'));
console.log(`PASS: ${cases} Three.js camera projections match existing orientation and depth; source-camera projection retained.`);
