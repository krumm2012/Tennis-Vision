const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context={};vm.createContext(context);vm.runInContext(fs.readFileSync(__dirname+'/coaching.js','utf8'),context);
const report=JSON.parse(fs.readFileSync(__dirname+'/coaching_report.json','utf8'));const meta={frames:250,fps:25,video_sha256:report.video_sha256};
assert.equal(context.TennisCoaching.validate(report,meta),report);
assert.throws(()=>context.TennisCoaching.validate(report,{...meta,video_sha256:'new-video'}));
assert.throws(()=>context.TennisCoaching.validate(report,{...meta,frames:20}));
const broken=structuredClone(report);broken.shots[0].phases[1].start=-1;assert.throws(()=>context.TennisCoaching.validate(broken,meta));
assert.equal(report.shots[0].phases.length,6);assert.equal(report.shots[2].complete,false);
console.log('PASS: coaching report video binding, phase bounds and incomplete shot');
