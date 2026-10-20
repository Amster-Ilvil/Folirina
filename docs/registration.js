/* Non-AI browser image registration.
 * Gradient corners + local normalized cross-correlation + 3-point affine RANSAC.
 * Unlike the legacy translation-only estimate, X and Y scaling are independent;
 * shear / small rotation / scanner drift can also be corrected. No remote API.
 */
function rgbaGray(image){
 const w=image.width,h=image.height, out=new Float32Array(w*h), d=image.data;
 for(let p=0,q=0;p<out.length;p++,q+=4){const alpha=d[q+3]/255;out[p]=alpha*(d[q]*.299+d[q+1]*.587+d[q+2]*.114)+(1-alpha)*255;}
 return {width:w,height:h,data:out};
}
export function downsampleGray(bitmap, ratio){
 const w=Math.max(40,Math.round(bitmap.width*ratio)),h=Math.max(40,Math.round(bitmap.height*ratio));
 const c=document.createElement('canvas');c.width=w;c.height=h;
 const cx=c.getContext('2d',{willReadFrequently:true});cx.fillStyle='white';cx.fillRect(0,0,w,h);cx.drawImage(bitmap,0,0,w,h);
 return rgbaGray(cx.getImageData(0,0,w,h));
}
function blur3(g){const {width:w,height:h,data:d}=g;const o=new Float32Array(d.length);
 for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){let i=y*w+x;o[i]=(d[i-w-1]+2*d[i-w]+d[i-w+1]+2*d[i-1]+4*d[i]+2*d[i+1]+d[i+w-1]+2*d[i+w]+d[i+w+1])/16;}
 for(let x=0;x<w;x++){o[x]=d[x];o[(h-1)*w+x]=d[(h-1)*w+x]}
 for(let y=0;y<h;y++){o[y*w]=d[y*w];o[y*w+w-1]=d[y*w+w-1]}
 return {...g,data:o};
}
function findCorners(g,limit=460){
 const {width:w,height:h,data:d}=g, gx=new Float32Array(d.length),gy=new Float32Array(d.length);
 for(let y=2;y<h-2;y++)for(let x=2;x<w-2;x++){let i=y*w+x;gx[i]=(d[i+1]-d[i-1])*.5;gy[i]=(d[i+w]-d[i-w])*.5;}
 const list=[];
 // Each 13x13 tile contributes at most its two strongest features.
 const tile=14,margin=16;
 for(let ty=margin;ty<h-margin;ty+=tile)for(let tx=margin;tx<w-margin;tx+=tile){
  let best=0,second=0,first=null,other=null;
  for(let y=ty;y<Math.min(ty+tile,h-margin);y+=3)for(let x=tx;x<Math.min(tx+tile,w-margin);x+=3){
   let xx=0,yy=0,xy=0;
   for(let v=-2;v<=2;v++)for(let u=-2;u<=2;u++){let k=(y+v)*w+x+u,a=gx[k],b=gy[k];xx+=a*a;yy+=b*b;xy+=a*b}
   const tr=xx+yy, eig=(tr-Math.sqrt((xx-yy)*(xx-yy)+4*xy*xy))*.5;
   if(eig>best){second=best;other=first;best=eig;first={x,y,score:eig}}
   else if(eig>second && (!first||Math.hypot(x-first.x,y-first.y)>7)){second=eig;other={x,y,score:eig}}
  }
  if(first)list.push(first);
  if(other&&second>best*.7)list.push(other);
 }
 list.sort((a,b)=>b.score-a.score);
 const result=[];const grid=new Set(),spacing=13;
 for(const p of list){if(p.score<1500)break;const cx=Math.floor(p.x/spacing),cy=Math.floor(p.y/spacing);let close=false;
  for(let j=-1;j<=1&&!close;j++)for(let i=-1;i<=1;i++)if(grid.has(`${cx+i}/${cy+j}`)){close=true;break}
  if(close)continue;grid.add(`${cx}/${cy}`);result.push(p);if(result.length>=limit)break;
 }
 return result;
}
function nccMatch(a,b,p,rad=65){
 const {width:aw,data:ad}=a,{width:bw,height:bh,data:bd}=b;
 const r=12,step=3;const samples=[];let sum=0,sumsq=0;
 for(let y=-r;y<=r;y+=step)for(let x=-r;x<=r;x+=step){let v=ad[(p.y+y)*aw+p.x+x];samples.push([x,y,v]);sum+=v;sumsq+=v*v}
 const n=samples.length,variance=sumsq-sum*sum/n;
 if(variance<28000)return null;
 const centerX=p.x*bw/aw,centerY=p.y*bh/a.height;
 const minX=Math.max(r+1,Math.round(centerX-rad)),maxX=Math.min(bw-r-2,Math.round(centerX+rad));
 const minY=Math.max(r+1,Math.round(centerY-rad)),maxY=Math.min(bh-r-2,Math.round(centerY+rad));
 let best=-1,px=0,py=0;
 const search=(x,y)=>{
  let bs=0,bss=0,cross=0;
  for(let j=0;j<samples.length;j++){
   let [u,v,t]=samples[j];const k=bd[(y+v)*bw+(x+u)];bs+=k;bss+=k*k;cross+=t*k;
  }
  let denom=(bss-bs*bs/n)*variance;
  const score=denom>1?((cross-sum*bs/n)/Math.sqrt(denom)):-1;
  if(score>best){best=score;px=x;py=y;}
 };
 for(let y=minY;y<=maxY;y+=2)for(let x=minX;x<=maxX;x+=2)search(x,y);
 const cx=px,cy=py;
 for(let y=Math.max(minY,cy-2);y<=Math.min(maxY,cy+2);y++)for(let x=Math.max(minX,cx-2);x<=Math.min(maxX,cx+2);x++)search(x,y);
 return best>=.68?{x:p.x,y:p.y,u:px,v:py,score:best}:null;
}
function transformFromTriplet(p,q,r){
 let det=p.x*(q.y-r.y)+q.x*(r.y-p.y)+r.x*(p.y-q.y);
 if(Math.abs(det)<75)return null;
 const aa=((q.y-r.y)*p.u+(r.y-p.y)*q.u+(p.y-q.y)*r.u)/det;
 const cc=((r.x-q.x)*p.u+(p.x-r.x)*q.u+(q.x-p.x)*r.u)/det;
 const ee=((q.x*r.y-r.x*q.y)*p.u+(r.x*p.y-p.x*r.y)*q.u+(p.x*q.y-q.x*p.y)*r.u)/det;
 const bb=((q.y-r.y)*p.v+(r.y-p.y)*q.v+(p.y-q.y)*r.v)/det;
 const dd=((r.x-q.x)*p.v+(p.x-r.x)*q.v+(q.x-p.x)*r.v)/det;
 const ff=((q.x*r.y-r.x*q.y)*p.v+(r.x*p.y-p.x*r.y)*q.v+(p.x*q.y-q.x*p.y)*r.v)/det;
 const det2=aa*dd-bb*cc;
 if(det2<.65||det2>1.55 || aa<.7||aa>1.4||dd<.7||dd>1.4||Math.abs(bb)>.20||Math.abs(cc)>.20) return null;
 return [aa,bb,cc,dd,ee,ff];
}
function solve3(m,v){
 const a=m.map((r,i)=>[...r,v[i]]);
 for(let i=0;i<3;i++){
  let b=i;for(let j=i+1;j<3;j++)if(Math.abs(a[j][i])>Math.abs(a[b][i]))b=j;
  if(Math.abs(a[b][i])<1e-8)return null;
  [a[b],a[i]]=[a[i],a[b]];
  let d=a[i][i];for(let k=i;k<4;k++)a[i][k]/=d;
  for(let j=0;j<3;j++)if(j!==i){let f=a[j][i];for(let k=i;k<4;k++)a[j][k]-=f*a[i][k];}
 }
 return a.map(z=>z[3]);
}
function refineAffine(points){
 let Sxx=0,Sxy=0,Sx=0,Syy=0,Sy=0,Su=0,Sxu=0,Syu=0,Sv=0,Sxv=0,Syv=0;
 for(const p of points){Sxx+=p.x*p.x;Sxy+=p.x*p.y;Sx+=p.x;Syy+=p.y*p.y;Sy+=p.y;Su+=p.u;Sxu+=p.x*p.u;Syu+=p.y*p.u;Sv+=p.v;Sxv+=p.x*p.v;Syv+=p.y*p.v;}
 const m=[[Sxx,Sxy,Sx],[Sxy,Syy,Sy],[Sx,Sy,points.length]];
 const x=solve3(m,[Sxu,Syu,Su]),y=solve3(m,[Sxv,Syv,Sv]);
 return x&&y?[x[0],y[0],x[1],y[1],x[2],y[2]]:null;
}
const squaredError=(t,p)=>{const dx=t[0]*p.x+t[2]*p.y+t[4]-p.u,dy=t[1]*p.x+t[3]*p.y+t[5]-p.v;return dx*dx+dy*dy};
export function fitAffineRansac(matches,srcWidth,srcHeight){
 if(matches.length<22)return null;
 let seed=12345678;
 const pick=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return Math.floor((seed/4294967296)*matches.length)};
 let best=null,count=0,score=-Infinity;
 for(let trial=0;trial<3600;trial++){
  let i=pick(),j=pick(),k=pick();if(i===j||j===k||i===k)continue;
  const t=transformFromTriplet(matches[i],matches[j],matches[k]);if(!t)continue;
  let n=0,s=0;
  for(const p of matches){const d=squaredError(t,p);if(d<5.76){n++;s+=Math.max(0,1-d/5.76)*p.score}}
  const fitness=n+s*.35;
  if(fitness>score){score=fitness;count=n;best=t;}
 }
 if(!best||count<18)return null;
 let inliers=[];
 for(let iter=0;iter<3;iter++){
  inliers=matches.filter(p=>squaredError(best,p)<(iter===0?9:6.25));
  if(inliers.length<18)return null;
  const refined=refineAffine(inliers);
  if(!refined)return null;best=refined;
 }
 inliers=matches.filter(p=>squaredError(best,p)<6.25);
 const xs=inliers.map(p=>p.x),ys=inliers.map(p=>p.y);
 const spanX=(Math.max(...xs)-Math.min(...xs))/srcWidth,spanY=(Math.max(...ys)-Math.min(...ys))/srcHeight;
 const median=(()=>{let z=inliers.map(p=>Math.sqrt(squaredError(best,p))).sort((a,b)=>a-b);return z[Math.floor(z.length/2)]||0})();
 if(inliers.length<18||spanX<.33||spanY<.3||median>2) return null;
 return {matrix:best,inliers:inliers.length,total:matches.length,medianError:median,coverage:[spanX,spanY]};
}
export async function registerAffineGrays(source,target,onProgress){
 const a=blur3(source),b=blur3(target),corners=findCorners(a,480),matches=[];
 for(let i=0;i<corners.length;i++){
  const hit=nccMatch(a,b,corners[i]);if(hit)matches.push(hit);
  if(i%18===17){onProgress?.(`匹配画面特征 ${i+1}/${corners.length}`);await new Promise(resolve=>setTimeout(resolve,0));}
 }
 const result=fitAffineRansac(matches,source.width,source.height);
 if(!result)throw Error(`特征匹配无法通过安全验证（匹配 ${matches.length} 处），请检查 SOURCE 配对页或使用桌面版`);
 // Reject narrow, ambiguous clusters (e.g. mostly text-only low-coverage pages).
 // A few good-looking matches are not proof of safe whole-page registration.
 if(result.inliers<35 || result.inliers/Math.max(1,matches.length)<.35)
   throw Error(`可靠配准证据不足（${result.inliers}/${matches.length} 个一致特征）；需要人工核对或桌面版`);
 return {...result,features:corners.length};
}
export async function estimateBrowserAffine(sourceBitmap,targetBitmap,onProgress){
 // Normalize EACH image independently; same absolute ratio makes matching
 // impossible between low-res Chinese scans and high-res color originals.
 const rs=Math.min(1,700/Math.max(sourceBitmap.width,sourceBitmap.height));
 const rt=Math.min(1,700/Math.max(targetBitmap.width,targetBitmap.height));
 const source=downsampleGray(sourceBitmap,rs),target=downsampleGray(targetBitmap,rt);
 const outcome=await registerAffineGrays(source,target,onProgress);
 const m=outcome.matrix;
 const ssx=source.width/sourceBitmap.width,ssy=source.height/sourceBitmap.height;
 const tsx=target.width/targetBitmap.width,tsy=target.height/targetBitmap.height;
 return {...outcome,matrix:[m[0]*ssx/tsx,m[1]*ssx/tsy,m[2]*ssy/tsx,m[3]*ssy/tsy,m[4]/tsx,m[5]/tsy]};
}
