const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context={};vm.createContext(context);vm.runInContext(fs.readFileSync(__dirname+'/mesh_renderer.js','utf8')+'\nthis.Renderer=SamMeshRenderer;',context);
const layout=(meta,image)=>JSON.parse(JSON.stringify(context.Renderer.textureLayout(meta,image)));
assert.deepEqual(layout({frames:250},{width:10240,height:5760}),{size:[1280,720],grid:[16,16],tile:[640,360]});
assert.deepEqual(layout({frames:90,image_size:[1920,1080],mask_atlas_grid:[10,9]},{width:4800,height:2430}),{size:[1920,1080],grid:[10,9],tile:[480,270]});
assert.throws(()=>layout({frames:91,mask_atlas_grid:[10,9]},{width:4800,height:2430}));
assert.throws(()=>layout({frames:1,image_size:[0,720]},{width:10240,height:5760}));
assert.throws(()=>layout({frames:1,mask_atlas_grid:[3,3]},{width:100,height:100}));
console.log('PASS: legacy and alternate texture layouts; invalid dimensions and atlas capacity rejected');
