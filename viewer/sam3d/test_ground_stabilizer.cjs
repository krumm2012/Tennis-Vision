const assert=require('node:assert/strict'),G=require('./ground_stabilizer.js');
const row=(i,y=0)=>({time:i/25,root:[0,y,0],feet:[0,1].map(k=>({local:[k*.2,-1+(i%2===k?.005:0),0],ground:[k*.2+(i%2?.003:-.003),0],camera:[k*.2,y,0]}))});
const rows=Array.from({length:40},(_,i)=>row(i));const result=G.solve(rows);
assert.equal(new Set(result.map(r=>r.support)).size,1,'small foot height noise must not switch support');
assert.deepEqual(G.solve(rows),result,'repeated playback/seek must be deterministic');
assert(Math.max(...result.map(r=>r.shift[0]))-Math.min(...result.map(r=>r.shift[0]))<.001,'planted foot remains stable');
const hop=[0,0,.04,.12,.16,.12,.04,0].map((y,i)=>({...row(i,y),airborne:y>0,contactConfidence:1})),flight=G.solve(hop);
assert(flight.some(r=>r.airborne));assert(flight[4].shift[1]>flight[1].shift[1]+.03,'inferred hop must retain elevation');assert(!flight.at(-1).airborne);
const step=Array.from({length:15},(_,i)=>{const r=row(i);r.feet[0].local[1]=i>3?-.8:-1.05;r.feet[1].local[1]=-1;return r;});
const switched=G.solve(step);assert.equal(switched[4].support,0);assert.equal(switched.at(-1).support,1);
console.log('PASS planted foot, support hysteresis, inferred hop, deterministic seek');

assert(!G.solve(hop.map(r=>({...r,contactConfidence:0}))).some(r=>r.airborne),'noisy camera roots cannot establish flight');
