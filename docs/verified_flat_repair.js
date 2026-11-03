/** 
 * Local browser adaptation of main/inpaint_preservation.py.
 * Pixel writes are confined to a human-reviewed erasure mask.
 */
export function preclearVerifiedFlat(out,original,mask,w,h){
 const n=w*h;
 if(!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||n>4000000||
    out.length!==n*4||original.length!==n*4||mask.length!==n)throw Error('局部纯色修复尺寸无效');
 const visited=new Uint8Array(n),cleared=new Uint8Array(n),stack=[];
 let components=0,pixels=0;
 for(let seed=0;seed<n;seed++){
  if(!mask[seed]||visited[seed])continue;
  stack.push(seed);visited[seed]=1;
  const island=[];let tooLarge=false;
  while(stack.length){
   const p=stack.pop(),x=p%w,y=(p/w)|0;
   if(island.length<2500)island.push(p);else tooLarge=true;
   for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dx&&!dy)continue;
    const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue;
    const q=yy*w+xx;if(mask[q]&&!visited[q]){visited[q]=1;stack.push(q)}
   }
  }
  if(tooLarge||island.length<3)continue;
  const ring=new Set();
  for(const p of island){
   const x=p%w,y=(p/w)|0;
   for(let dy=-8;dy<=8;dy++)for(let dx=-8;dx<=8;dx++){
    if(dx*dx+dy*dy>64)continue;
    const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue;
    const q=yy*w+xx;
    if(!mask[q]&&original[q*4+3]>=245)ring.add(q);
   }
  }
  if(ring.size<32)continue;
  const ids=[...ring],samples=[[],[],[]];
  for(const p of ids)for(let c=0;c<3;c++)samples[c].push(original[p*4+c]);
  const med=[];
  let spread=0;
  for(const chan of samples){
   chan.sort((a,b)=>a-b);
   med.push(chan[Math.floor(chan.length/2)]);
   spread=Math.max(spread,chan[Math.floor((chan.length-1)*.95)]-chan[Math.floor((chan.length-1)*.05)]);
  }
  if(spread>8)continue;
  let matching=0;
  for(const p of ids){
   const j=p*4;
   if(Math.abs(original[j]-med[0])<=6&&Math.abs(original[j+1]-med[1])<=6&&Math.abs(original[j+2]-med[2])<=6)matching++;
  }
  if(matching/ids.length<.99)continue;
  for(const p of island){
   const j=p*4;for(let c=0;c<3;c++)out[j+c]=med[c];
   out[j+3]=original[j+3];cleared[p]=1;pixels++;
  }
  components++;
 }
 return {mask:cleared,components,pixels};
}
