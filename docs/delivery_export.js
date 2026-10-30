/**
 * Browser-only publication encoding, ported from the *options* introduced by
 * desktop src/manga_hd_transfer/publication.py on 2026-10-30.
 * Unlike desktop Caesium/Upscayl, this uses native browser codecs, performs no
 * super-resolution, never resizes the TARGET, and never changes edit layers.
 */
import {canvasBlob} from './core.js';

const formats=Object.freeze({png:{mime:'image/png',extension:'png'},webp:{mime:'image/webp',extension:'webp'},jpeg:{mime:'image/jpeg',extension:'jpg'}});
export function deliverySettings(format='png',quality=90){
 if(!Object.hasOwn(formats,format))throw Error('不支持的导出格式');
 const q=Number(quality);
 if(!Number.isInteger(q)||q<1||q>100)throw Error('导出质量应为 1–100');
 return {format,quality:q,...formats[format]};
}
export function exportFilename(name,spec){
 const stem=String(name||'page').replace(/\.[^.]+$/,'').replace(/[\\/:*?"<>|\x00-\x1f]/g,'_')||'page';
 return `${stem}.${spec.extension}`;
}
export async function encodeDeliveryCanvas(canvas,spec){
 if(!canvas||canvas.width<1||canvas.height<1)throw Error('没有可导出的图像');
 let output=canvas;
 // JPEG cannot retain transparency; composite explicitly over white rather
 // than silently getting black/unpredictable transparent pixels.
 if(spec.format==='jpeg'){
  output=document.createElement('canvas');output.width=canvas.width;output.height=canvas.height;
  const ctx=output.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,output.width,output.height);ctx.drawImage(canvas,0,0);
 }
 try{
  const blob=await canvasBlob(output,spec.mime,spec.quality/100);
  // Some browser implementations silently return PNG for unsupported formats.
  if(blob.type!==spec.mime)throw Error(`当前浏览器不支持 ${spec.extension.toUpperCase()} 编码，请选择 PNG`);
  return blob;
 }finally{if(output!==canvas){output.width=0;output.height=0}}
}
export async function encodeDeliveryPngBlob(pngBlob,spec){
 if(spec.format==='png')return pngBlob;
 const bitmap=await createImageBitmap(pngBlob);
 try{
  const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
  canvas.getContext('2d').drawImage(bitmap,0,0);
  try{return await encodeDeliveryCanvas(canvas,spec)}finally{canvas.width=0;canvas.height=0}
 }finally{bitmap.close?.()}
}
