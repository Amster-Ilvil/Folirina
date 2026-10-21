/* No model, no OpenCV runtime: bounded 4-point RANSAC with least-squares
   projective refinement. Coordinates are normalized to [0,1] to avoid
   ill-conditioned 8x8 systems. Only selected when substantially better than
   the affine baseline across the entire page. */
const finite=a=>a.every(Number.isFinite);
export function solveLinear(a,b){
 const n=b.length,mat=a.map((r,i)=>[...r,b[i]]);
 for(let k=0;k<n;k++){
  let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(mat[i][k])>Math.abs(mat[pivot][k]))pivot=i;
  if(Math.abs(mat[pivot][k])<1e-10)return null;
  [mat[k],mat[pivot]]=[mat[pivot],mat[k]];
  const v=mat[k][k];for(let j=k;j<=n;j++)mat[k][j]/=v;
  for(let i=0;i<n;i++)if(i!==k){const scale=mat[i][k];if(!scale)continue;for(let j=k;j<=n;j++)mat[i][j]-=scale*mat[k][j]}
 }
 return mat.map(r=>r[n]);
}
function equations(pts,sw,sh,tw,th){
 const a=[],b=[];
 for(const p of pts){const x=p.x/sw,y=p.y/sh,u=p.u/tw,v=p.v/th;
  a.push([x,y,1,0,0,0,-x*u,-y*u]);b.push(u);
  a.push([0,0,0,x,y,1,-x*v,-y*v]);b.push(v);
 }return [a,b];
}
function fit(pts,sw,sh,tw,th){
 if(pts.length<4)return null;
 const [a,b]=equations(pts,sw,sh,tw,th),n=8;
 const gram=Array.from({length:n},()=>Array(n).fill(0)),rhs=Array(n).fill(0);
 for(let k=0;k<a.length;k++)for(let i=0;i<n;i++){
  rhs[i]+=a[k][i]*b[k];for(let j=i;j<n;j++)gram[i][j]+=a[k][i]*a[k][j];
 }
 for(let i=0;i<n;i++)for(let j=0;j<i;j++)gram[i][j]=gram[j][i];
 const z=solveLinear(gram,rhs);
 return z&&finite(z)?[z[0]*tw/sw,z[1]*tw/sh,z[2]*tw, z[3]*th/sw,z[4]*th/sh,z[5]*th,z[6]/sw,z[7]/sh,1]:null;
}
export function mapH(h,x,y){const z=h[6]*x+h[7]*y+h[8];if(Math.abs(z)<1e-9)return [NaN,NaN];return [(h[0]*x+h[1]*y+h[2])/z,(h[3]*x+h[4]*y+h[5])/z];}
export function invertH(h){
 const [a,b,c,d,e,f,g,j,k]=h;
 const A=e*k-f*j,B=c*j-b*k,C=b*f-c*e;
 const D=f*g-d*k,E=a*k-c*g,F=c*d-a*f;
 const G=d*j-e*g,H=b*g-a*j,I=a*e-b*d;
 const det=a*A+b*D+c*G;if(Math.abs(det)<1e-10)return null;
 return [A,B,C,D,E,F,G,H,I].map(v=>v/det);
}
function reasonable(h,sw,sh,tw,th){
 // Reject folding, explosive perspective and geometry outside the page scale.
 let prev=null;
 for(const [x,y] of [[0,0],[sw,0],[sw,sh],[0,sh]]){
  const z=h[6]*x+h[7]*y+1;
  if(z<.7||z>1.5)return false;
  const p=mapH(h,x,y);if(!finite(p)||Math.abs(p[0])>tw*4||Math.abs(p[1])>th*4)return false;
  if(prev&&Math.hypot(p[0]-prev[0],p[1]-prev[1])<Math.min(tw,th)*.15)return false;
  prev=p;
 }
 const pts=[[0,0],[sw,0],[sw,sh],[0,sh]].map(p=>mapH(h,...p));let sign=0;
 for(let i=0;i<4;i++){
  const [x0,y0]=pts[i],[x1,y1]=pts[(i+1)%4],[x2,y2]=pts[(i+2)%4];const cross=(x1-x0)*(y2-y1)-(y1-y0)*(x2-x1);
  if(Math.abs(cross)<tw*th*.02||sign&&Math.sign(cross)!==sign)return false;sign=Math.sign(cross);
 }
 return true;
}
function errorSq(h,p){const [u,v]=mapH(h,p.x,p.y);return (u-p.u)**2+(v-p.v)**2;}
function median(arr){const x=arr.slice().sort((a,b)=>a-b);return x[Math.floor(x.length/2)]??Infinity;}
export function fitHomography(matches,sw,sh,tw,th,affine){
 if(matches.length<40)return null;
 let seed=2654435761;const pick=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return Math.floor(seed/4294967296*n)};
 let best=null,bestWeight=0;
 for(let trial=0;trial<1600;trial++){
  const ids=new Set();while(ids.size<4)ids.add(pick(matches.length));
  const subset=[...ids].map(k=>matches[k]);
  const h=fit(subset,sw,sh,tw,th);if(!h||!reasonable(h,sw,sh,tw,th))continue;
  let score=0;
  for(const p of matches){const e=errorSq(h,p);if(e<7.84)score+=1+Math.max(0,1-e/7.84)*p.score*.25;}
  if(score>bestWeight){bestWeight=score;best=h;}
 }
 if(!best)return null;
 let inliers=[];
 for(let k=0;k<3;k++){
  inliers=matches.filter(p=>errorSq(best,p)<(k===0?10:7.84));
  if(inliers.length<30)return null;
  const f=fit(inliers,sw,sh,tw,th);if(!f||!reasonable(f,sw,sh,tw,th))return null;best=f;
 }
 inliers=matches.filter(p=>errorSq(best,p)<7.84);
 if(inliers.length<35||inliers.length/Math.max(1,matches.length)<.30)return null;
 const xs=inliers.map(p=>p.x),ys=inliers.map(p=>p.y);
 if((Math.max(...xs)-Math.min(...xs))/sw<.38||(Math.max(...ys)-Math.min(...ys))/sh<.35)return null;
 if(!affine){
  // No affine model: demand exceptional geometric evidence to avoid mistaking
  // a wrong page pair for strong perspective.
  const med=median(inliers.map(p=>Math.sqrt(errorSq(best,p))));
  if(inliers.length<55||inliers.length/matches.length<.55||med>1.1||
     (Math.max(...xs)-Math.min(...xs))/sw<.5||
     (Math.max(...ys)-Math.min(...ys))/sh<.5)return null;
  return {matrix:best,medianError:med,affineMedianError:null,inliers:inliers.length};
 }
 // Compare BOTH models on their mutually accepted correspondences; models
 // must not cherry-pick different pages or locally convenient text features.
 const evaluate=inliers.filter(p=>{
  const ax=affine[0]*p.x+affine[2]*p.y+affine[4];
  const ay=affine[1]*p.x+affine[3]*p.y+affine[5];
  return (ax-p.u)**2+(ay-p.v)**2<49;
 });
 if(evaluate.length<30)return null;
 const errP=median(evaluate.map(p=>Math.sqrt(errorSq(best,p))));
 const errA=median(evaluate.map(p=>Math.hypot(affine[0]*p.x+affine[2]*p.y+affine[4]-p.u,affine[1]*p.x+affine[3]*p.y+affine[5]-p.v)));
 // A perspective warp with no *material* improvement only adds failure risk.
 if(!Number.isFinite(errP)||!Number.isFinite(errA)||errP>.95*errA||errA-errP<.16||errP>2.2)return null;
 return {matrix:best,medianError:errP,affineMedianError:errA,inliers:inliers.length};
}
export function toFullResolution(h,sw,sh,tw,th,srcW,srcH,dstW,dstH){
 const s=[sw/srcW,sh/srcH],t=[tw/dstW,th/dstH];
 const n=[h[0]*s[0]/t[0],h[1]*s[1]/t[0],h[2]/t[0],h[3]*s[0]/t[1],h[4]*s[1]/t[1],h[5]/t[1],h[6]*s[0],h[7]*s[1],1];
 return finite(n)?n:null;
}
