const fs=require('fs'),vm=require('vm'),assert=require('assert');
const p=__dirname,html=fs.readFileSync(p+'/viewer.html','utf8'),shader=fs.readFileSync(p+'/mesh_renderer.js','utf8');
// Execute the actual JS camera/project functions and the scalar GLSL expressions.
const camera=html.match(/function camera\(p\)\{([^}]+)\}/)[1];
const project=html.match(/function project\(p\)\{([^}]+)\}/)[1];
const scalar=shader.match(/float x=p.x\*cy[^;]+;/)[0].replace('float ','let ');
function calc(P,yaw,pitch){let c={P,yaw,pitch,center:[0,0,0],sub:(a,b)=>a.map((x,i)=>x-b[i]),r:{width:640,height:450},scale:100};vm.createContext(c);return vm.runInContext(`const js=(function(p){${camera}})(P);const screen=(function(p){${project}})(P);const p={x:P[0],y:P[1],z:P[2]},cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);${scalar}({js,screen,gpu:[x,y,depth],clipZ:-depth/20})`,c)}
for(const yaw of [0,.3,Math.PI])for(const pitch of [-.5,0,.2,.8])for(const P of [[1,2,3],[-1,0,1],[0,1,-1]]){const t=calc(P,yaw,pitch);t.js.forEach((v,i)=>assert(Math.abs(v-t.gpu[i])<1e-12));assert(Math.abs(t.screen[0]-(320+t.gpu[0]*100))<1e-10);assert(Math.abs(t.screen[1]-(225-t.gpu[1]*100))<1e-10)}
assert(calc([0,0,1],0,0).clipZ<calc([0,0,-1],0,0).clipZ,'camera side is nearer');
assert(calc([0,0,-1],Math.PI,0).clipZ<calc([0,0,1],Math.PI,0).clipZ,'mirror side is nearer');
assert(calc([0,1,0],0,.2).clipZ<calc([0,-1,0],0,.2).clipZ,'positive pitch observes from above');
assert(calc([-1,0,0],0,0).screen[0]<320);assert(calc([-1,0,0],Math.PI,0).screen[0]>320);
assert(shader.includes(',-depth/20.,1.)'));assert(html.includes('.sort((a,b)=>a.depth-b.depth)'));
assert(shader.includes('if(sourcePass==1){gl_Position=sourceClip(originalCamera);return;}'));assert(shader.includes('gl_Position=sourceClip(reflected);return;'));assert(shader.includes('(f+n)/(f-n)*p.z-2.*f*n/(f-n),p.z)')); // Source depth retains its perspective convention.
assert(html.includes("$('frontView').onclick=()=>{yaw=0;"));assert(html.includes("$('backView').onclick=()=>{yaw=Math.PI;"));
console.log('PASS: actual 2D/3D view equations agree; front/back occlusion, positive elevation, horizontal identity, presets; source shaders unchanged.');
