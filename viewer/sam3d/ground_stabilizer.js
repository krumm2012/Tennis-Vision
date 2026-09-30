// Precompute in timestamp order: seeking backwards must produce the same pose.
class GroundStabilizer {
 static solve(rows){
  const norm=a=>Math.hypot(...a),sub=(a,b)=>a.map((x,k)=>x-b[k]);
  let support=0,pending=-1,pendingSince=0,previous=null,anchor=null,airborne=false,takeoff=null;
  return rows.map((row,i)=>{
   const dt=i?row.time-rows[i-1].time:0,reset=!i||dt<=0||dt>.12;
   const low=row.feet[0].local[1]<=row.feet[1].local[1]?0:1;
   if(reset){support=low;pending=-1;previous=null;anchor=null;airborne=false;}
   const speed=row.feet.map((f,k)=>reset?0:norm(sub(f.camera,rows[i-1].feet[k].camera))/dt);
   // Monocular camera-root jitter is not reliable evidence of flight. Only
   // honor flight when an upstream estimator explicitly supplies confidence.
   const flight=row.airborne===true&&row.contactConfidence>=.8;
   if(flight&&!airborne){airborne=true;takeoff={root:reset?row.root[1]:rows[i-1].root[1],y:previous?.[1]??-row.feet[low].local[1]};anchor=null;}
   if(!flight&&airborne){airborne=false;anchor=null;}
   if(low!==support&&row.feet[support].local[1]-row.feet[low].local[1]>.025){
    if(pending!==low){pending=low;pendingSince=row.time;}
    if(row.time-pendingSince>=.08){support=low;pending=-1;anchor=null;}
   }else pending=-1;
   const foot=row.feet[support],contact=!airborne&&speed[support]<.65;
   if(!contact)anchor=null;
   if(contact&&!anchor)anchor=foot.ground.slice();
   // A planted foot keeps its ground point; release when it moves or observation
   // drifts beyond 8 cm, so a real step cannot remain pinned indefinitely.
   if(anchor&&Math.hypot(anchor[0]-foot.ground[0],anchor[1]-foot.ground[1])>.08)anchor=foot.ground.slice();
   const ground=anchor||foot.ground,target=[ground[0]-foot.local[0],-foot.local[1],ground[1]-foot.local[2]];
   if(airborne)target[1]=takeoff.y+Math.max(0,Math.min(1,row.root[1]-takeoff.root));
   const distance=previous?norm(sub(target,previous)):0;
   const alpha=reset?1:1-Math.exp(-dt*(8+Math.min(20,distance/Math.max(dt,.001)*4)));
   const shift=previous?previous.map((v,k)=>v+(target[k]-v)*alpha):target;
   previous=shift;
   return {shift,support,contact,airborne};
  });
 }
}
if(typeof module!=='undefined')module.exports=GroundStabilizer;
