import {repairDirectionalTarget} from './structure_repair.js';
/**
 * Fully client-side, explicitly HUMAN-APPROVED text masks.
 * No automatic destructive authority: only separately reviewed source-writing
 * and target-erasing masks are used. Original TARGET remains immutable.
 */
export function encodeBitMask(values){
 const data=new Uint8Array(Math.ceil(values.length/8));
 for(let i=0;i<values.length;i++)if(values[i])data[i>>3]|=1<<(i&7);
 let result='';for(let i=0;i<data.length;i+=8192)result+=String.fromCharCode(...data.subarray(i,i+8192));
 return btoa(result);
}
export function decodeBitMask(data,n){
 if(typeof data!=='string'||!Number.isInteger(n)||n<1||n>4_000_000)throw Error('掩码长度不合法');
 const bin=atob(data),length=Math.ceil(n/8);
 if(bin.length!==length)throw Error('掩码编码与选区尺寸不匹配');
 const mask=new Uint8Array(n);
 for(let i=0;i<n;i++)mask[i]=(bin.charCodeAt(i>>3)>>(i&7))&1;
 return mask;
}
export function reviewedEdit(draft,repairMode='neighbors'){
 if(!draft||!draft.box)throw Error('没有可确认的审阅掩码');
 const {box,sourceMask,targetMask}=draft;
 if(box.w*box.h>4_000_000||sourceMask.length!==box.w*box.h||targetMask.length!==box.w*box.h)throw Error('掩码尺寸无效');
 if(!sourceMask.some(Boolean)||!targetMask.some(Boolean))throw Error('SOURCE 与 TARGET 掩码不能留空');
 if(!['neighbors','source','directional'].includes(repairMode))throw Error('不支持的背景恢复方式');
 return {kind:'reviewed',shape:'brush',box:{...box},repairMode,
   sourceBits:encodeBitMask(sourceMask),targetBits:encodeBitMask(targetMask)};
}
/** Repair mask pixels using unmapped original TARGET neighbor evidence.
 * For structured scenes, 2-axis interpolation is only a candidate for review,
 * not perfect OpenCV Telea or semantic inpainting. Keeps edits pixel-local.
 */
export function applyReviewedTransfer(current,original,source,edit,w,h){
 const n=w*h;
 if(!source||source.length!==n*4||original.length!==n*4||current.length!==n*4)throw Error('掩码源图/底图尺寸不匹配');
 const sourceMask=decodeBitMask(edit.sourceBits,n),targetMask=decodeBitMask(edit.targetBits,n);
 const next=new Uint8ClampedArray(current);
 let cleared=0,written=0,noDonor=0,lineBridges=0,uncertain=0;
 if(edit.repairMode==='directional'){
   const stats=repairDirectionalTarget(next,original,targetMask,w,h);
   ({cleared,noDonor,lineBridges,uncertain}=stats);
 }
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const p=y*w+x;if(!targetMask[p]||edit.repairMode==='directional')continue;
   const j=p*4;
   if(edit.repairMode==='source' && source[j+3]>=240 && !sourceMask[p]){
     for(let c=0;c<3;c++)next[j+c]=source[j+c];
     cleared++;continue;
   }
   const donors=[];
   // Sample opposing sides without crossing the editable target mask. Using
   // only original (immutable) pixels avoids propagating previously filled RGB.
   for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){
     for(let step=1;step<=22;step++){
       const xx=x+dx*step,yy=y+dy*step;if(xx<0||xx>=w||yy<0||yy>=h)break;
       const q=yy*w+xx;if(targetMask[q])continue;
       const k=q*4;donors.push([step,original[k],original[k+1],original[k+2]]);break;
     }
   }
   if(!donors.length){noDonor++;continue}
   let sum=0,rs=0,gs=0,bs=0;
   for(const [distance,r,g,b] of donors){const weight=1/Math.max(1,distance);sum+=weight;rs+=r*weight;gs+=g*weight;bs+=b*weight}
   next[j]=Math.round(rs/sum);next[j+1]=Math.round(gs/sum);next[j+2]=Math.round(bs/sum);cleared++;
 }
 // Paint reviewed SOURCE ink last, so translation never disappears under erase.
 for(let p=0;p<n;p++){
  if(!sourceMask[p])continue;
  const j=p*4;if(source[j+3]<240)continue;
  next[j]=source[j];next[j+1]=source[j+1];next[j+2]=source[j+2];next[j+3]=255;written++;
 }
 let changed=0;for(let p=0;p<n;p++){const j=p*4;if(next[j]!==current[j]||next[j+1]!==current[j+1]||next[j+2]!==current[j+2])changed++}
 current.set(next);
 return {applied:changed>0,changed,cleared,written,noDonor,lineBridges,uncertain,method:'reviewed-source-target-masks'};
}
