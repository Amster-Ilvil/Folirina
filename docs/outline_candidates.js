/**
 * Bounded preview-only candidate finder for dark TARGET glyphs with white
 * outlines on coloured art. Conservative JS adaptation of desktop main's
 * outlined_text_mask.py (2026-10-30). Detection is NOT erase authority.
 */
const dirs=[-1,0,1];
export function outlinedTargetCandidates(rgba,selection,w,h){
 if(rgba.length!==w*h*4||selection.length!==w*h*4)throw Error('白描边辅助图尺寸不匹配');
 if(w*h>4_000_000)throw Error('请缩小白描边检查范围');
 const n=w*h,white=new Uint8Array(n),seen=new Uint8Array(n),mask=new Uint8Array(n);
 const dark=new Uint8Array(n);
 for(let p=0;p<n;p++){
  const j=p*4,r=rgba[j],g=rgba[j+1],b=rgba[j+2];
  if(selection[j+3]<128||rgba[j+3]<245)continue;
  white[p]=(Math.min(r,g,b)>=235&&Math.max(r,g,b)-Math.min(r,g,b)<=24)?1:0;
  dark[p]=(.299*r+.587*g+.114*b)<150?1:0;
 }
 let accepted=0,components=0,rejected=0;
 const stack=[],points=[];
 const neutralCandidates=[];
 const paintCandidate=({minX,minY,maxX,maxY,bw,bh,component,outside})=>{
  // Only the inspected selection may supply write candidates.
  for(let yy=Math.max(0,minY-3);yy<=Math.min(h-1,maxY+3);yy++)
   for(let xx=Math.max(0,minX-3);xx<=Math.min(w-1,maxX+3);xx++){
    const p=yy*w+xx;if(selection[p*4+3]<128)continue;
    let close=false;
    for(let dy=-3;dy<=3&&!close;dy++)for(let dx=-3;dx<=3;dx++){
     if(dx*dx+dy*dy>9)continue;
     const x=xx+dx,y=yy+dy;if(x<minX||x>maxX||y<minY||y>maxY)continue;
     const q=(y-minY)*bw+x-minX;
     if(component[q]||!outside[q]){close=true;break}
    }
    if(close&&!mask[p]){mask[p]=1;accepted++}
   }
  components++;
 };

 for(let start=0;start<n;start++){
  if(!white[start]||seen[start])continue;
  seen[start]=1;stack.push(start);points.length=0;
  let minX=w,maxX=0,minY=h,maxY=0;
  while(stack.length){
   const p=stack.pop(),x=p%w,y=(p/w)|0;points.push(p);
   if(x<minX)minX=x;if(x>maxX)maxX=x;if(y<minY)minY=y;if(y>maxY)maxY=y;
   for(const dy of dirs)for(const dx of dirs){
    if(!dx&&!dy)continue;
    const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue;
    const q=yy*w+xx;if(!white[q]||seen[q])continue;seen[q]=1;stack.push(q);
   }
  }
  const bw=maxX-minX+1,bh=maxY-minY+1,area=points.length;
  // Anything touching the crop edge or wide enough to be speech paper is
  // explicitly out of scope. Never recruit an entire white speech balloon.
  if(area<8||bw>192||bh>192||bw*bh>12000||minX===0||minY===0||maxX===w-1||maxY===h-1){rejected++;continue}
  const nn=bw*bh,component=new Uint8Array(nn),outside=new Uint8Array(nn),todo=[];
  for(const p of points)component[((((p/w)|0)-minY)*bw+(p%w)-minX)]=1;
  // Flood-fill the inverse white component from its bounding-box border.
  for(let yy=0;yy<bh;yy++)for(let xx=0;xx<bw;xx++){
   if(xx!==0&&yy!==0&&xx!==bw-1&&yy!==bh-1)continue;
   const q=yy*bw+xx;if(component[q]||outside[q])continue;outside[q]=1;todo.push(q);
  }
  while(todo.length){const p=todo.pop(),x=p%bw,y=(p/bw)|0;
   for(const dy of dirs)for(const dx of dirs){
    if(!dx&&!dy)continue;const xx=x+dx,yy=y+dy;
    if(xx<0||xx>=bw||yy<0||yy>=bh)continue;
    const q=yy*bw+xx;if(!component[q]&&!outside[q]){outside[q]=1;todo.push(q)}
   }
  }
  let filledArea=0,darkCount=0;
  const seeds=[],distance=new Int16Array(nn);distance.fill(-1);
  for(let yy=0;yy<bh;yy++)for(let xx=0;xx<bw;xx++){
   const q=yy*bw+xx,global=(minY+yy)*w+(minX+xx),isHole=!component[q]&&!outside[q];
   if(component[q]||isHole)filledArea++;
   if(isHole&&dark[global]){darkCount++;distance[q]=0;seeds.push(q)}
  }
  if(darkCount<3||darkCount/Math.max(1,filledArea)<.015){rejected++;continue}
  // Approximate L2 distance conservatively via 8-connected steps: false
  // positives are rejected even when the true outline is a little wider.
  let front=0;
  while(front<seeds.length){const q=seeds[front++],x=q%bw,y=(q/bw)|0;
   if(distance[q]>=13)continue;
   for(const dy of dirs)for(const dx of dirs){
    if(!dx&&!dy)continue;const xx=x+dx,yy=y+dy;
    if(xx<0||xx>=bw||yy<0||yy>=bh)continue;
    const pos=yy*bw+xx;if(distance[pos]!==-1)continue;
    distance[pos]=distance[q]+1;seeds.push(pos);
   }
  }
  let near=0;for(let q=0;q<nn;q++)if(component[q]&&distance[q]>=0&&distance[q]<=12)near++;
  if(near/Math.max(1,area)<.95){rejected++;continue}
  const xa=Math.max(0,minX-12),ya=Math.max(0,minY-12),xb=Math.min(w,maxX+13),yb=Math.min(h,maxY+13);
  let colored=0,total=0;
  for(let y=ya;y<yb;y++)for(let x=xa;x<xb;x++){
   const p=y*w+x,j=p*4,inside=x>=minX&&x<=maxX&&y>=minY&&y<=maxY;
   if(inside&&component[(y-minY)*bw+x-minX])continue;
   total++;
   const r=rgba[j],g=rgba[j+1],b=rgba[j+2];
   if(Math.max(r,g,b)>=120&&Math.max(r,g,b)-Math.min(r,g,b)>=30)colored++;
  }
  if(colored<Math.max(24,Math.ceil(total*.3))){
   // Neutral black/white manga: unlike coloured paper, accept only a
   // manually scoped *run* of at least three aligned white-outlined glyphs.
   let neutral=0,context=0,selectedFill=0,filled=0;
   const levels=[];
   for(let yy=ya;yy<yb;yy++)for(let xx=xa;xx<xb;xx++){
    const p=yy*w+xx,j=p*4,inside=xx>=minX&&xx<=maxX&&yy>=minY&&yy<=maxY;
    if(inside&&component[(yy-minY)*bw+(xx-minX)])continue;
    const r=rgba[j],g=rgba[j+1],b=rgba[j+2];
    if(Math.max(r,g,b)-Math.min(r,g,b)<=24)neutral++;
    levels.push(Math.round(.299*r+.587*g+.114*b));context++;
   }
   for(let yy=minY;yy<=maxY;yy++)for(let xx=minX;xx<=maxX;xx++){
    const q=(yy-minY)*bw+xx-minX;
    if(!component[q]&&outside[q])continue;
    filled++;if(selection[(yy*w+xx)*4+3]>=128)selectedFill++;
   }
   levels.sort((a,b)=>a-b);
   const median=levels[Math.floor(levels.length/2)]??255;
   if(context>=24&&neutral/context>=.8&&median<=170&&filled>=8&&selectedFill/filled>=.98&&neutralCandidates.length<400)
    neutralCandidates.push({minX,minY,maxX,maxY,bw,bh,component,outside});
   else rejected++;
   continue;
  }
  paintCandidate({minX,minY,maxX,maxY,bw,bh,component,outside});
 }
 // Reject isolated white highlights: neutral candidates only qualify as
 // a connected run of three or more similarly sized aligned glyphs.
 const graph=Array.from({length:neutralCandidates.length},()=>[]);
 for(let i=0;i<neutralCandidates.length;i++){
  const a=neutralCandidates[i];
  for(let j=0;j<i;j++){
   const b=neutralCandidates[j];
   if(Math.max(a.bw,b.bw)/Math.max(1,Math.min(a.bw,b.bw))>2||
      Math.max(a.bh,b.bh)/Math.max(1,Math.min(a.bh,b.bh))>2)continue;
   const dx=Math.abs((a.minX+a.bw/2)-(b.minX+b.bw/2));
   const dy=Math.abs((a.minY+a.bh/2)-(b.minY+b.bh/2));
   const connected=(dy<=Math.max(a.bh,b.bh)*.5&&dx>.4*Math.min(a.bw,b.bw)&&dx<=2.5*Math.max(a.bw,b.bw))||
    (dx<=Math.max(a.bw,b.bw)*.5&&dy>.4*Math.min(a.bh,b.bh)&&dy<=2.5*Math.max(a.bh,b.bh));
   if(connected){graph[i].push(j);graph[j].push(i)}
  }
 }
 const visitedNeutral=new Uint8Array(neutralCandidates.length);
 for(let i=0;i<neutralCandidates.length;i++){
  if(visitedNeutral[i])continue;
  const pending=[i],group=[];visitedNeutral[i]=1;
  while(pending.length){
   const k=pending.pop();group.push(k);
   for(const j of graph[k])if(!visitedNeutral[j]){visitedNeutral[j]=1;pending.push(j)}
  }
  if(group.length<3){rejected+=group.length;continue}
  for(const k of group)paintCandidate(neutralCandidates[k]);
 }
 return {mask,accepted,components,rejected};
}