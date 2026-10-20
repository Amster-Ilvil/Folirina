/* Folirina Web Lite. Non-AI port informed by Folirina's page_pairing, registration and direct_patch modules.
   Target is always the immutable original base. No original pixels are overwritten. */
export const isImageName = s => /\.(png|jpe?g|webp|bmp|gif)$/i.test(s);
export const naturalCompare = (a,b) => a.localeCompare(b,undefined,{numeric:true,sensitivity:'base'});
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const safeName=s=>String(s).replace(/[\\/<>:"|?*\x00-\x1f]/g,'_').replace(/^\.+/,'').slice(0,150)||'page.png';
const cache = new Map();
export function clearBitmaps(){for(const b of cache.values()){if(b.bitmap?.close)b.bitmap.close()}cache.clear()}
export async function bitmap(item){
 if(!item) return null;
 const k=item.id;
 if(cache.has(k)){const entry=cache.get(k);cache.delete(k);cache.set(k,entry);return entry.promise}
 const promise=createImageBitmap(item.blob,{imageOrientation:'from-image'});
 cache.set(k,{promise,bitmap:null});
 try {const b=await promise;const v=cache.get(k);if(v)v.bitmap=b;while(cache.size>5){let [old,e]=cache.entries().next().value;cache.delete(old);if(e.bitmap?.close)e.bitmap.close()}return b}
 catch(err){cache.delete(k);throw err}
}
export async function unpack(files,prefix){
 const out=[];let seq=0;
 for(const f of files){
   if(/\.(zip|cbz)$/i.test(f.name)){
     const z=await JSZip.loadAsync(f);
     const entries=Object.values(z.files).filter(e=>!e.dir&&!/(^|\/)__MACOSX\//i.test(e.name)&&!/(^|\/)\._/.test(e.name)&&isImageName(e.name)).sort((a,b)=>naturalCompare(a.name,b.name));
     if(entries.length>1000)throw Error('单个 ZIP 最多导入 1000 张图片');
     for(const e of entries){if(e._data?.uncompressedSize>60*1024*1024)throw Error('ZIP 内有超过 60MB 的图片，请分卷导入');const blob=await e.async('blob');out.push({id:`${prefix}-${++seq}-${Date.now()}`,name:e.name,blob,url:URL.createObjectURL(blob)})}
   }else if(isImageName(f.name))out.push({id:`${prefix}-${++seq}-${Date.now()}`,name:f.webkitRelativePath||f.name,blob:f,url:URL.createObjectURL(f)});
 }
 return out.sort((a,b)=>naturalCompare(a.name,b.name));
}
export function releaseItems(items){items.forEach(i=>URL.revokeObjectURL(i.url));clearBitmaps()}
export function defaultPair(targetIndex,sourceCount){return {tgt:targetIndex,src:targetIndex<sourceCount?targetIndex:null,skip:false,dx:0,dy:0,scale:1,affine:null,alignDiagnostics:null,edits:[],undo:[],redo:[],score:null}}
export function nativeCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c}
export function brushOrSelectionMask(edit,x,y,w,h){
 const c=nativeCanvas(w,h),cx=c.getContext('2d');cx.fillStyle='#fff';cx.strokeStyle='#fff';cx.lineCap='round';cx.lineJoin='round';
 if(edit.shape==='ellipse'||edit.shape==='rect'){
   const l=Math.min(edit.start.x,edit.end.x)-x,t=Math.min(edit.start.y,edit.end.y)-y;
   const rw=Math.abs(edit.end.x-edit.start.x),rh=Math.abs(edit.end.y-edit.start.y);
   const guard=Math.min(4,Math.min(rw,rh)*.11);
   if(rw<=guard*2||rh<=guard*2)return cx.getImageData(0,0,w,h).data;
   if(edit.shape==='ellipse'){cx.beginPath();cx.ellipse(l+rw/2,t+rh/2,rw/2-guard,rh/2-guard,0,0,Math.PI*2);cx.fill()}
   else cx.fillRect(l+guard,t+guard,rw-guard*2,rh-guard*2);
 }else{
   const p=edit.points||[];cx.lineWidth=edit.brush||20;
   if(p.length===1){cx.beginPath();cx.arc(p[0].x-x,p[0].y-y,cx.lineWidth/2,0,Math.PI*2);cx.fill()}
   else if(p.length>1){cx.beginPath();cx.moveTo(p[0].x-x,p[0].y-y);p.slice(1).forEach(k=>cx.lineTo(k.x-x,k.y-y));cx.stroke()}
 }
 return cx.getImageData(0,0,w,h).data;
}
export function getBounds(edit,W,H){
 const brush=edit.shape==='brush';
 const pts=brush?(edit.points||[]):[edit.start,edit.end].filter(Boolean);if(!pts.length)return null;
 const r=brush?((edit.brush||20)/2+3):2;
 const x=clamp(Math.floor(Math.min(...pts.map(p=>p.x))-r),0,W),y=clamp(Math.floor(Math.min(...pts.map(p=>p.y))-r),0,H);
 const xx=clamp(Math.ceil(Math.max(...pts.map(p=>p.x))+r),0,W),yy=clamp(Math.ceil(Math.max(...pts.map(p=>p.y))+r),0,H);
 return xx>x&&yy>y?{x,y,w:xx-x,h:yy-y}:null;
}
export function warpedSource(source,targetWidth,targetHeight,pair){
 const c=nativeCanvas(targetWidth,targetHeight);const ctx=c.getContext('2d',{willReadFrequently:true});
 if(source){
  if(Array.isArray(pair.affine)&&pair.affine.length===6&&pair.affine.every(Number.isFinite)){
   const [a,b,c,d,e,f]=pair.affine,zoom=pair.scale||1;
   const midX=targetWidth/2,midY=targetHeight/2;
   ctx.setTransform(a*zoom,b*zoom,c*zoom,d*zoom,midX+zoom*(e-midX)+(pair.dx||0),midY+zoom*(f-midY)+(pair.dy||0));
   ctx.drawImage(source,0,0);ctx.resetTransform();
  }else{
   // Compatibility with workspaces created before non-AI affine registration.
   const s=targetWidth/source.width*(pair.scale||1);
   ctx.drawImage(source,pair.dx||0,pair.dy||0,source.width*s,source.height*s);
  }
 }
 return c;
}
export function compose(target,source,pair,settings={}){
 if(!target)throw Error('缺少 TARGET');
 const W=target.width,H=target.height;const canvas=nativeCanvas(W,H);const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.drawImage(target,0,0);
 if(!pair.edits?.length)return canvas;
 const base=nativeCanvas(W,H),baseCtx=base.getContext('2d',{willReadFrequently:true});baseCtx.drawImage(target,0,0);
 const srcC=warpedSource(source,W,H,pair),srcCtx=srcC.getContext('2d',{willReadFrequently:true});
 for(const edit of pair.edits){
   const box=getBounds(edit,W,H);if(!box)continue;
   const mask=brushOrSelectionMask(edit,box.x,box.y,box.w,box.h);
   const current=ctx.getImageData(box.x,box.y,box.w,box.h),dst=current.data;
   const original=baseCtx.getImageData(box.x,box.y,box.w,box.h).data;
   const src=edit.kind==='transfer'?srcCtx.getImageData(box.x,box.y,box.w,box.h).data:null;
   const threshold=edit.threshold??settings.threshold??185;
   const paper=edit.paper??settings.paper??255;
   const offset=edit.cloneOffset||{x:0,y:0};
   const origFull=edit.kind==='clone'?baseCtx.getImageData(0,0,W,H).data:null;
   for(let y=0;y<box.h;y++)for(let x=0;x<box.w;x++){
     const i=(y*box.w+x)*4;if(mask[i+3]<128)continue;
     if(edit.kind==='transfer'){
       if(!src||src[i+3]<1)continue;
       const v=(src[i]*.299+src[i+1]*.587+src[i+2]*.114);
       if(edit.mode==='transparent'){
         // Preview/stored semi-transparent source ink, never erase Japanese or expand the mask.
         if(v>threshold)continue;
         const alpha=.46;
         dst[i]=Math.round(dst[i]*(1-alpha)+src[i]*alpha);
         dst[i+1]=Math.round(dst[i+1]*(1-alpha)+src[i+1]*alpha);
         dst[i+2]=Math.round(dst[i+2]*(1-alpha)+src[i+2]*alpha);
       }else if(v<=threshold){dst[i]=src[i];dst[i+1]=src[i+1];dst[i+2]=src[i+2]}
       else{dst[i]=paper;dst[i+1]=paper;dst[i+2]=paper}
     }else if(edit.kind==='erase'){dst[i]=paper;dst[i+1]=paper;dst[i+2]=paper}
     else if(edit.kind==='restore'){dst[i]=original[i];dst[i+1]=original[i+1];dst[i+2]=original[i+2]}
     else if(edit.kind==='clone'){
       const sx=clamp(Math.round(box.x+x+offset.x),0,W-1),sy=clamp(Math.round(box.y+y+offset.y),0,H-1),j=(sy*W+sx)*4;
       dst[i]=origFull[j];dst[i+1]=origFull[j+1];dst[i+2]=origFull[j+2];
     }
     dst[i+3]=255;
   }
   ctx.putImageData(current,box.x,box.y);
 }
 return canvas;
}
export function alignmentScore(source,target,dx=0,dy=0){
 const w=95,h=Math.round(95*target.height/target.width),a=nativeCanvas(w,h),b=nativeCanvas(w,h);
 const ac=a.getContext('2d',{willReadFrequently:true}),bc=b.getContext('2d',{willReadFrequently:true});
 const sx=w/target.width,sy=h/target.height;
 ac.filter='blur(2px)';ac.drawImage(target,0,0,w,h);bc.filter='blur(2px)';bc.drawImage(source,dx*sx,dy*sy,w,source.height*(w/source.width));
 const p=ac.getImageData(0,0,w,h).data,q=bc.getImageData(0,0,w,h).data;
 let n=0,sum=0;
 for(let y=6;y<h-6;y+=2)for(let x=6;x<w-6;x+=2){const i=(y*w+x)*4;if(q[i+3]<250)continue;const u=(p[i]+p[i+1]+p[i+2])/3,v=(q[i]+q[i+1]+q[i+2])/3;sum+=Math.min(1,Math.abs(u-v)/180);n++}
 return n?sum/n:1;
}
export function estimateTranslation(source,target,priorX=0,priorY=0){
 if(!source||!target)return {dx:0,dy:0,score:1};const step=Math.max(2,Math.round(target.width/60));let best={dx:priorX,dy:priorY,score:alignmentScore(source,target,priorX,priorY)};
 const span=Math.max(2,Math.round(target.width*.045/step));
 for(let y=-span;y<=span;y++)for(let x=-span;x<=span;x++){
   const dx=priorX+x*step,dy=priorY+y*step,score=alignmentScore(source,target,dx,dy);
   if(score<best.score)best={dx,dy,score};
 }
 const local=Math.max(1,Math.round(step/3)),cx=best.dx,cy=best.dy;for(let y=-3;y<=3;y++)for(let x=-3;x<=3;x++){
   const dx=cx+x*local,dy=cy+y*local,score=alignmentScore(source,target,dx,dy);
   if(score<best.score)best={dx,dy,score};
 }
 return best;
}
export async function structureFingerprint(item){const b=await bitmap(item);const c=nativeCanvas(28,40),cx=c.getContext('2d',{willReadFrequently:true});cx.filter='blur(2px)';cx.drawImage(b,0,0,28,40);const d=cx.getImageData(0,0,28,40).data;const out=new Float32Array(28*40);for(let i=0;i<out.length;i++)out[i]=(d[i*4]*.299+d[i*4+1]*.587+d[i*4+2]*.114)/255;return out}
export function fingerprintDistance(a,b){let s=0,n=0;for(let y=2;y<38;y++)for(let x=2;x<26;x++){const i=y*28+x;s+=Math.abs(a[i]-b[i]);n++}return s/n}
export async function smartPairs(sources,targets,pairs,onProgress){
 const srcF=[],tgtF=[];
 for(let i=0;i<sources.length;i++){srcF.push(await structureFingerprint(sources[i]));onProgress?.(`计算中文版特征 ${i+1}/${sources.length}`)}
 for(let i=0;i<targets.length;i++){tgtF.push(await structureFingerprint(targets[i]));onProgress?.(`计算日文版特征 ${i+1}/${targets.length}`)}
 const used=new Set(),out=pairs.map(p=>({...p}));
 for(let i=0;i<out.length;i++){
   const candidates=[];for(let j=Math.max(0,i-5);j<Math.min(sources.length,i+6);j++)if(!used.has(j))candidates.push({src:j,score:fingerprintDistance(srcF[j],tgtF[i])+Math.abs(i-j)*.004});
   candidates.sort((a,b)=>a.score-b.score);
   const best=candidates[0];if(best){out[i].src=best.src;out[i].score=best.score;used.add(best.src)}else{out[i].src=null;out[i].score=null}
 }
 return out;
}
export async function canvasBlob(canvas,type='image/png',quality=1){return new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('无法编码图像')),type,quality))}
export function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),20_000)}
