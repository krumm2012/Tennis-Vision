const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const read=name=>fs.readFileSync(path.join(__dirname,name),'utf8');
function fn(source,name){const start=source.indexOf('function '+name+'(');let end=source.indexOf('{',start),depth=1;while(depth){end++;if(source[end]==='{')depth++;if(source[end]==='}')depth--;}return (source.slice(start-6,start)==='async '?'async ':'')+source.slice(start,end+1);}
const source=read('viewer.html');
let failures=0;
function test(name,run){try{run();console.log('PASS',name)}catch(e){failures++;console.error('FAIL',name,e.message)}}
for(const file of ['calibration_editor.js','racket_editor.js'])test(file+' cancels buffer before editing',()=>{
 const stop=Error('stop after pause'),ctx={data:{},v:{readyState:4,pause(){assert.equal(ctx.bufferingPlayback,false,'editing must cancel automatic resume');throw stop}},bufferingPlayback:true,$:()=>({})};
 vm.createContext(ctx);if(source.includes('function pausePlayback('))vm.runInContext(fn(source,'pausePlayback'),ctx);
 vm.runInContext(fn(read(file),'open'),ctx);try{ctx.open()}catch(e){if(e!==stop)throw e}
});
test('seek does not render until decoded video is ready',()=>{
 const ctx={data:{frames:[{time:0},{time:1}]},n:0,bufferingPlayback:true,$:()=>({}),v:{pause(){},set currentTime(t){this.seeking=true}},render(){assert.equal(ctx.v.seeking,false,'rendered during seek')}};
 vm.createContext(ctx);if(source.includes('function pausePlayback('))vm.runInContext(fn(source,'pausePlayback'),ctx);vm.runInContext(fn(source,'jump'),ctx);ctx.jump(1);assert.equal(ctx.n,1);
});
process.exitCode=failures?1:0;
(async()=>{
 const elements=new Map(),ctx={meshLoading:false,meshError:null,bufferingPlayback:true,n:0,v:{pause(){}},$:id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id)},render(){ctx.renders++},renders:0};
 let attempts=0;ctx.meshRenderer={ready:false,async load(){if(++attempts===1)throw Error('offline');this.ready=true}};
 vm.createContext(ctx);for(const name of ['pausePlayback','meshFailed','loadMesh'])vm.runInContext(fn(source,name),ctx);
 try{await ctx.loadMesh();assert.equal(ctx.meshError.message,'offline');assert.equal(ctx.bufferingPlayback,false);assert.equal(ctx.$('retryMesh').hidden,false);await ctx.loadMesh();assert.equal(ctx.meshError,null);assert.equal(ctx.meshRenderer.ready,true);assert.equal(ctx.$('retryMesh').hidden,true);assert.equal(ctx.renders,1);console.log('PASS: failed initial mesh load can retry successfully')}catch(e){failures++;console.error('FAIL retry',e)}
 process.exitCode=failures?1:0;
})();
