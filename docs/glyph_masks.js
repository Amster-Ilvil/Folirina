/**
 * Experimental, non-destructive dual ink mask inspection.
 * Inspired by Folirina desktop text_transfer.py: independent SOURCE and TARGET
 * candidate masks; preserve structures present in both editions.
 * Deliberately does NOT claim semantic/OCR classification or erase anything.
 */
const maskClamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const luma=(r,g,b)=>Math.round(.299*r+.587*g+.114*b);

function localInk(rgba,w,h,radius=5,delta=19){
 const sum=new Float64Array((w+1)*(h+1));
 const gray=new Uint8Array(w*h);
 for(let y=0;y<h;y++){
  let line=0;
  for(let x=0;x<w;x++){
   const p=y*w+x,j=p*4;
   gray[p]=luma(rgba[j],rgba[j+1],rgba[j+2]);
   line+=gray[p];sum[(y+1)*(w+1)+x+1]=sum[y*(w+1)+x+1]+line;
  }
 }
 const ink=new Uint8Array(w*h);
 for(let y=0;y<h;y++){
  const y0=Math.max(0,y-radius),y1=Math.min(h,y+radius+1);
  for(let x=0;x<w;x++){
   const x0=Math.max(0,x-radius),x1=Math.min(w,x+radius+1),p=y*w+x;
   if(rgba[p*4+3]<245)continue;
   const mean=(sum[y1*(w+1)+x1]-sum[y0*(w+1)+x1]-sum[y1*(w+1)+x0]+sum[y0*(w+1)+x0])/((x1-x0)*(y1-y0));
   if(mean-gray[p]>=delta)ink[p]=1;
  }
 }
 return ink;
}

function nearby(mask,w,h,x,y,r=1){
 const xa=Math.max(0,x-r),xb=Math.min(w-1,x+r),ya=Math.max(0,y-r),yb=Math.min(h-1,y+r);
 for(let yy=ya;yy<=yb;yy++)for(let xx=xa;xx<=xb;xx++)if(mask[yy*w+xx])return true;
 return false;
}

function candidateComponents(ink,other,selection,w,h){
 const used=new Uint8Array(w*h),result=new Uint8Array(w*h),stack=[];
 let rejectedArea=0,acceptedArea=0,components=0;
 const dirs=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]];
 for(let start=0;start<w*h;start++){
  if(!ink[start]||used[start]||selection[start*4+3]<128)continue;
  used[start]=1;stack.push(start);
  const points=[];let unique=0,minX=w,minY=h,maxX=0,maxY=0;
  while(stack.length){
   const p=stack.pop(),x=p%w,y=Math.floor(p/w);points.push(p);
   if(!nearby(other,w,h,x,y,1))unique++;
   minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);
   for(const [dx,dy] of dirs){
    const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue;
    const q=yy*w+xx;if(used[q]||!ink[q]||selection[q*4+3]<128)continue;
    used[q]=1;stack.push(q);
   }
  }
  const area=points.length,bw=maxX-minX+1,bh=maxY-minY+1;
  const fraction=area/Math.max(1,w*h);
  const longLine=(bw>.80*w||bh>.80*h)&&area/Math.max(1,bw*bh)<.26;
  // Require edition-specific evidence, reject massive/background-wide regions.
  const accept=area>=3&&unique>=3&&unique/area>=.34&&fraction<.07&&!longLine;
  if(accept){for(const p of points)result[p]=1;acceptedArea+=area;components++}
  else rejectedArea+=area;
 }
 return {mask:result,acceptedArea,rejectedArea,components};
}

export function inspectTextDifferences(source,target,selection,w,h,opts={}){
 if(!source||!target||source.length!==w*h*4||target.length!==w*h*4||selection.length!==w*h*4)throw Error('字形掩码输入尺寸不匹配');
 const delta=maskClamp(opts.delta??19,8,60),radius=maskClamp(Math.round(opts.radius??5),2,16);
 const a=localInk(source,w,h,radius,delta),b=localInk(target,w,h,radius,delta);
 const sourceCandidate=candidateComponents(a,b,selection,w,h);
 const targetCandidate=candidateComponents(b,a,selection,w,h);
 let shared=0,selected=0,possible=0;
 for(let p=0;p<w*h;p++){
  if(selection[p*4+3]<128)continue;
  selected++;
  if(a[p]&&nearby(b,w,h,p%w,Math.floor(p/w),1))shared++;
  if(a[p]||b[p])possible++;
 }
 const structuralSupport=shared/Math.max(1,possible),
   candidateRatio=(sourceCandidate.acceptedArea+targetCandidate.acceptedArea)/Math.max(1,possible);
 // This score is diagnostic only, never permission for destructive changes.
 const caution=selected<64?'选区太小':structuralSupport<.16?'两版公共线条较少，配准或版面差异可能较大':candidateRatio>.72?'大量像素呈现差异，可能包含背景网点和建筑结构':'仍可能误判建筑线条、网点或彩色纹理，必须人工检查';
 return {sourceMask:sourceCandidate.mask,targetMask:targetCandidate.mask,width:w,height:h,
   summary:{selected,shared,possible,structuralSupport,sourceCandidates:sourceCandidate.acceptedArea,
    targetCandidates:targetCandidate.acceptedArea,sourceComponents:sourceCandidate.components,
    targetComponents:targetCandidate.components,caution,destructiveSafe:false}};
}

export function maskOverlay(inspection,w,h){
 const rgba=new Uint8ClampedArray(w*h*4);
 for(let i=0;i<w*h;i++){
  const src=inspection.sourceMask[i],tgt=inspection.targetMask[i];if(!src&&!tgt)continue;
  const j=i*4;
  if(src&&tgt){rgba[j]=252;rgba[j+1]=189;rgba[j+2]=35}
  else if(tgt){rgba[j]=246;rgba[j+1]=55;rgba[j+2]=80}
  else {rgba[j]=27;rgba[j+1]=222;rgba[j+2]=116}
  rgba[j+3]=200;
 }
 return rgba;
}
