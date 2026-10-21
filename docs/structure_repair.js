/**
 * Optional, browser-only directional TARGET restoration for an explicitly
 * reviewed Japanese erasure mask. It does NOT segment glyphs or approve masks.
 * Opposite edge samples can reconnect a narrow manga rule across removed ink;
 * otherwise it falls back to the conservative four-ray local interpolation.
 * No reads from a previous repaired pixel: all evidence comes from immutable TARGET.
 */
const DIRS=[[1,0],[0,1],[1,1],[1,-1],[1,2],[2,1],[1,-2],[2,-1]];
const CARDINALS=[[1,0],[-1,0],[0,1],[0,-1]];
const lum=(a,b,c)=>.299*a+.587*b+.114*c;

export function repairDirectionalTarget(out, original, mask, w, h, maxRay=22){
 if(out.length!==w*h*4||original.length!==w*h*4||mask.length!==w*h)throw Error('定向修复像素与掩码尺寸不一致');
 if(!Number.isInteger(maxRay)||maxRay<4||maxRay>64)throw Error('定向修复搜索半径不合法');
 let cleared=0,noDonor=0,lineBridges=0,uncertain=0;
 const lookup=(x,y,dx,dy)=>{
  for(let step=1;step<=maxRay;step++){
   const xx=x+dx*step,yy=y+dy*step;
   if(xx<0||xx>=w||yy<0||yy>=h)break;
   const p=yy*w+xx;if(!mask[p])return {x:xx,y:yy,j:p*4,d:step};
  }
  return null;
 };
 const perpendicularContrast=(v,dx,dy)=>{
  const norm=Math.hypot(dx,dy);
  const px=Math.round(-dy/norm*3),py=Math.round(dx/norm*3),base=lum(original[v.j],original[v.j+1],original[v.j+2]);
  let bright=0,valid=0;
  for(const sign of [-1,1]){
   const xx=v.x+px*sign,yy=v.y+py*sign;
   if(xx<0||xx>=w||yy<0||yy>=h||mask[yy*w+xx])continue;
   const j=(yy*w+xx)*4;
   bright+=lum(original[j],original[j+1],original[j+2])-base;valid++;
  }
  return valid===2?bright/2:0;
 };
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const p=y*w+x;if(!mask[p])continue;
  const j=p*4, options=[];
  const donor=[];
  for(const [dx,dy] of CARDINALS){const v=lookup(x,y,dx,dy);if(v)donor.push(v)}
  for(const [dx,dy] of DIRS){
   const a=lookup(x,y,dx,dy),b=lookup(x,y,-dx,-dy);if(!a||!b)continue;
   const la=lum(original[a.j],original[a.j+1],original[a.j+2]);
   const lb=lum(original[b.j],original[b.j+1],original[b.j+2]);
   const diff=(Math.abs(original[a.j]-original[b.j])+Math.abs(original[a.j+1]-original[b.j+1])+Math.abs(original[a.j+2]-original[b.j+2]))/3;
   // Positive thin-rule evidence on both sides of the erasure, not merely two
   // dark samples at unrelated text corners.
   const continued=(v,sign)=>{
     const base=lum(original[v.j],original[v.j+1],original[v.j+2]);
     for(let i=1;i<=2;i++){
       const xx=v.x+dx*sign*i,yy=v.y+dy*sign*i;
       if(xx<0||xx>=w||yy<0||yy>=h||mask[yy*w+xx])return false;
       const k=(yy*w+xx)*4,here=lum(original[k],original[k+1],original[k+2]);
       if(here>160||Math.abs(here-base)>45)return false;
     }
     return true;
   };
   const bridge=la<130&&lb<130&&perpendicularContrast(a,dx,dy)>48&&perpendicularContrast(b,dx,dy)>48
      &&continued(a,+1)&&continued(b,-1)&&diff<=28;
   const weight=bridge?115:0;
   const score=diff + (a.d+b.d)*.65 - weight;
   options.push({a,b,score,diff,bridge});
  }
  options.sort((a,b)=>a.score-b.score);
  const chosen=options[0];
  if(chosen?.bridge){
   const a=chosen.a,b=chosen.b;
   for(let c=0;c<3;c++)out[j+c]=Math.round((original[a.j+c]*b.d+original[b.j+c]*a.d)/(a.d+b.d));
   if(chosen.bridge)lineBridges++;
   cleared++;
  }else if(donor.length){
   let sum=0,channels=[0,0,0];
   for(const d of donor){const weight=1/d.d;sum+=weight;for(let c=0;c<3;c++)channels[c]+=original[d.j+c]*weight}
   for(let c=0;c<3;c++)out[j+c]=Math.round(channels[c]/sum);
   cleared++;uncertain++;
  }else noDonor++;
  out[j+3]=255;
 }
 return {cleared,noDonor,lineBridges,uncertain};
}
