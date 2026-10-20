/* Folirina Web Lite v0.2.5: deterministic flat-background protection.
 * Never use a white clear on a colorful target. Complex/art backgrounds
 * are guarded rather than silently destroyed by automatic transparent mode.
 * No segmentation model or OCR is involved.
 */
export const luminance=(r,g,b)=>.299*r+.587*g+.114*b;
const sq=n=>n*n;
const dist=(a,b)=>Math.sqrt(sq(a[0]-b[0])+sq(a[1]-b[1])+sq(a[2]-b[2]));
const cap=(x,a,b)=>Math.max(a,Math.min(b,x));

export function samplePaper(data,mask,{stride=1,maxSamples=6000}={}){
  const n=mask.length>>2;
  const step=Math.max(stride,Math.floor(n/maxSamples));
  const samples=[];
  const histogram=new Map();
  for(let i=0;i<n;i+=step){
    const j=i*4;if(mask[j+3]<128||data[j+3]<128)continue;
    const rgb=[data[j],data[j+1],data[j+2]];
    // Group neighboring RGB values, not just luma: colored paper stays colored.
    const bin=((rgb[0]>>4)<<8)|((rgb[1]>>4)<<4)|(rgb[2]>>4);
    samples.push(rgb);
    histogram.set(bin,(histogram.get(bin)||0)+1);
  }
  if(samples.length<12)return {safe:false,reason:'选区太小，不能确定背景',color:[255,255,255],coverage:0};
  const sorted=[...histogram.entries()].sort((a,b)=>b[1]-a[1]);
  // Choose the most common paper-like cluster. For white manga this is near
  // white, not necessarily the brightest pixel of a colored scene.
  let best=sorted[0][0];
  let max=0;
  for(const [bin,count] of sorted.slice(0,6)){
    const r=(bin>>8&15)*16+8,g=(bin>>4&15)*16+8,b=(bin&15)*16+8;
    const weighted=count*(luminance(r,g,b)>=125?1.05:0.4);
    if(weighted>max){max=weighted;best=bin;}
  }
  const anchor=[((best>>8)&15)*16+8,((best>>4)&15)*16+8,(best&15)*16+8];
  const cluster=samples.filter(p=>dist(p,anchor)<=31);
  const coverage=cluster.length/samples.length;
  const color=[0,1,2].map(i=>{
    const chan=cluster.map(p=>p[i]).sort((a,b)=>a-b);
    return chan.length?chan[Math.floor(chan.length/2)]:anchor[i];
  });
  const lum=luminance(...color);
  const distinct=Math.max(...color)-Math.min(...color);
  // More than ~36% of a brush ROI deviating from one coherent background is
  // usually text over art/screentones: do not auto-paint it as flat paper.
  const safe=coverage>=.64 && lum>=125;
  return {safe,color,coverage,luminance:lum,kind:distinct>15?'color':'mono',reason:safe?'':`纹理较复杂（背景一致度 ${Math.round(coverage*100)}%），自动清除已跳过`};
}

export function applyProtectedInk(dst,original,src,mask,profileTarget,profileSource,mode='transparent',threshold=185){
  const n=dst.length>>2;
  const paper=profileTarget.color;
  const srcPaperLum=profileSource.luminance;
  if(!profileTarget.safe || !profileSource.safe){
    // When the source/target cannot be parsed as flat paper, preserve TARGET.
    // Hole is intentionally also protected; use the explicit erase/clone tools
    // for user-controlled artwork backgrounds.
    return {applied:false,reason:(!profileTarget.safe?profileTarget:profileSource).reason};
  }
  let edited=0;
  for(let k=0;k<n;k++){
    const j=k*4;
    if(mask[j+3]<128||src[j+3]<128)continue;
    const sr=src[j],sg=src[j+1],sb=src[j+2];
    const sl=luminance(sr,sg,sb),tl=luminance(original[j],original[j+1],original[j+2]);
    const diff=srcPaperLum-sl;
    // Black/dark source lettering on lighter flat paper (including chromatic
    // flats). Use alpha from ink contrast for anti-aliased letter edges.
    const alpha=diff>24 && (sl<=threshold || diff>53)?cap((diff-24)/50,0,1):0;
    const targetDeviation=dist([original[j],original[j+1],original[j+2]],paper);
    const oldInk=(profileTarget.luminance-tl>19 && targetDeviation>27);
    const beforeR=dst[j],beforeG=dst[j+1],beforeB=dst[j+2];
    if(mode==='hole' || oldInk){
      dst[j]=paper[0];dst[j+1]=paper[1];dst[j+2]=paper[2];
    }
    if(alpha>.02){
      // Preserve the original SOURCE ink color, not a fixed translucent gray.
      dst[j]=Math.round(dst[j]*(1-alpha)+sr*alpha);
      dst[j+1]=Math.round(dst[j+1]*(1-alpha)+sg*alpha);
      dst[j+2]=Math.round(dst[j+2]*(1-alpha)+sb*alpha);
    }
    if(Math.abs(dst[j]-beforeR)+Math.abs(dst[j+1]-beforeG)+Math.abs(dst[j+2]-beforeB)>2)edited++;
    dst[j+3]=255;
  }
  return {applied:true,changed:edited,color:paper,kind:profileTarget.kind};
}
