import {unpack,releaseItems,bitmap,defaultPair,compose,smartPairs,canvasBlob,download,naturalCompare,safeName,warpedSource} from './core.js';
import {estimateBrowserAffine} from './registration.js';
import {getBounds,brushOrSelectionMask} from './core.js';
import {inspectTextDifferences,maskOverlay} from './glyph_masks.js';

const $=id=>document.getElementById(id);
const state={sources:[],targets:[],pairs:[],index:0,view:'result',tool:'paint-transfer',brush:24,threshold:185,paper:255,gesture:null,inspection:null,cloneAnchor:null,stamp:0,rendering:false,hover:null,transferMode:'hole'};
const cvs=$('editorCanvas'),over=$('overlayCanvas'),ctx=cvs.getContext('2d',{willReadFrequently:true}),oc=over.getContext('2d');
let toastHandle=null;
// Optional, local-only native backend. GitHub Pages always remains fully Lite.
const nativeEngine={ready:false,enabled:false,token:null};
const nativeData=new WeakMap();
function dataURL(item){
 if(!item)throw Error('没有已导入的图片');
 if(nativeData.has(item))return nativeData.get(item);
 const promise=new Promise((resolve,reject)=>{
   const rd=new FileReader();rd.onload=()=>resolve(rd.result);rd.onerror=()=>reject(Error('图片读取失败'));rd.readAsDataURL(item.blob)
 });nativeData.set(item,promise);return promise;
}
async function nativePost(op,data){
 if(!nativeEngine.ready||!nativeEngine.token)throw Error('原生引擎未连接');
 const res=await fetch('/api/engine/'+op,{method:'POST',credentials:'same-origin',
   headers:{'Content-Type':'application/json','X-Folirina-Token':nativeEngine.token},
   body:JSON.stringify(data)});
 const out=await res.json();if(!res.ok||out.error)throw Error(out.error||'原生引擎操作失败');return out;
}
async function nativeRegister(sourceItem,targetItem){
 return nativePost('register',{source:await dataURL(sourceItem),target:await dataURL(targetItem)});
}
function nativeEffectiveMatrix(pair,tgtWidth,tgtHeight){
 if(!Array.isArray(pair.nativeMatrix))throw Error('当前页没有原生透视配准矩阵，请重新执行自动配准');
 const H=pair.nativeMatrix,z=pair.scale||1;
 const shiftX=tgtWidth*(1-z)*.5+(pair.dx||0),shiftY=tgtHeight*(1-z)*.5+(pair.dy||0);
 return [
  H[0].map((v,i)=>z*v+shiftX*H[2][i]),
  H[1].map((v,i)=>z*v+shiftY*H[2][i]),H[2].slice()
 ];
}
async function nativeRender(pair,srcItem,tgtItem){
 const target=await bitmap(tgtItem);
 const matrix=Array.isArray(pair.nativeMatrix)?nativeEffectiveMatrix(pair,target.width,target.height):
   (pair.edits.some(e=>e.kind==='transfer')?null:[[1,0,0],[0,1,0],[0,0,1]]);
 if(!matrix)throw Error('当前页未完成原生配准，不能迁移文字');
 return nativePost('render',{source:await dataURL(srcItem||tgtItem),target:await dataURL(tgtItem),matrix,
   edits:pair.edits.map(e=>({kind:e.kind,mode:e.mode,shape:e.shape,start:e.start,end:e.end,
    points:e.points,brush:e.brush,paper:e.paper,cloneOffset:e.cloneOffset}))});
}
async function nativeWarp(pair,srcItem,tgtItem){
 const target=await bitmap(tgtItem);
 return nativePost('warp',{source:await dataURL(srcItem),target:await dataURL(tgtItem),matrix:nativeEffectiveMatrix(pair,target.width,target.height)});
}
function decodedPNG(data){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(Error('原生 PNG 解码失败'));img.src=data});}
async function engineInit(){
 try{
   const res=await fetch('/api/engine/health',{cache:'no-store'});
   if(!res.ok)throw Error('not local');const info=await res.json();if(!info.ok||!info.token)throw Error('no local engine');
   nativeEngine.ready=true;nativeEngine.enabled=true;nativeEngine.token=info.token;
   $('engineToggle').disabled=false;$('engineToggle').textContent='引擎：原生 Python';
   document.querySelector('[data-transfer-mode=precision]').disabled=false;
   $('engineStatus').textContent='已连接本机原生引擎';
   $('engineStatus').title='本机 OpenCV 配准 + 原版 Folirina 区域函数，所有图片仅发送到 127.0.0.1';
 }catch(_){nativeEngine.ready=false;nativeEngine.enabled=false;document.querySelector('[data-transfer-mode=precision]').disabled=true;
   $('engineStatus').textContent='免安装 Lite';
   $('engineToggle').title='完整版：在安装了 Folirina 的电脑运行 python tools/web_bridge.py，访问 http://127.0.0.1:8765/';}
}
$('engineToggle').addEventListener('click',()=>{
 if(!nativeEngine.ready)return;
 if(state.pairs.some(p=>p.edits.length)&&!confirm('切换引擎会改变现有编辑的合成效果。建议先保存工作区，确认切换？'))return;
 nativeEngine.enabled=!nativeEngine.enabled;
 document.querySelector('[data-transfer-mode=precision]').disabled=!nativeEngine.enabled;
 if(!nativeEngine.enabled&&state.transferMode==='precision'){state.transferMode='hole';document.querySelectorAll('[data-transfer-mode]').forEach(b=>b.classList.toggle('active',b.dataset.transferMode==='hole'));}
 $('engineToggle').textContent='引擎：'+(nativeEngine.enabled?'原生 Python':'浏览器 Lite');
 for(const p of state.pairs){if(!p.edits.length){p.affine=null;p.homography=null;p.nativeMatrix=null;p.alignDiagnostics=null;p.alignError=null;}}
 toast(nativeEngine.enabled?'原生引擎模式：配准及区域合成使用桌面 Python 代码':'已返回纯浏览器 Lite 模式');
 updateUI();
});
// GitHub Pages is a self-contained browser tool. Keep the optional native
// engine available ONLY when a user intentionally opens the local server.
// Public visitors should not see a disabled Python-only processing option.
const runningLocal=/^(localhost|127\.0\.0\.1)$/.test(location.hostname);
if(runningLocal){void engineInit();}
else{
 for(const el of [$('engineToggle'),$('engineStatus'),document.querySelector('.engine-help'),document.querySelector('[data-transfer-mode="precision"]')]){
   if(el)el.style.display='none';
 }
}


// Display transform is independent of document edits. Never recompute fit on a stroke.
const zoomState={mode:'fit',percent:100,pageId:null};
function applyZoom({resetCenter=false}={}){
 const sc=$('canvasScroller');if(!cvs.width||!cvs.height||sc.classList.contains('hidden'))return;
 const cw=Math.max(10,sc.clientWidth),ch=Math.max(10,sc.clientHeight);
 const oldWidth=parseFloat(cvs.style.width)||cvs.width,oldHeight=parseFloat(cvs.style.height)||cvs.height;
 const relX=(sc.scrollLeft+cw/2)/oldWidth,relY=(sc.scrollTop+ch/2)/oldHeight;
 let factor=zoomState.mode==='fit'?Math.min((cw-76)/cvs.width,(ch-60)/cvs.height):
   zoomState.mode==='width'?(cw-76)/cvs.width:zoomState.percent/100;
 factor=Math.max(.05,Math.min(3,factor));
 const width=Math.max(1,Math.round(cvs.width*factor)),height=Math.max(1,Math.round(cvs.height*factor));
 cvs.style.width=over.style.width=$('canvasHolder').style.width=width+'px';
 cvs.style.height=over.style.height=$('canvasHolder').style.height=height+'px';
 if(resetCenter){sc.scrollLeft=Math.max(0,(width-cw)/2);sc.scrollTop=0}
 else{sc.scrollLeft=Math.max(0,relX*width-cw/2);sc.scrollTop=Math.max(0,relY*height-ch/2)}
 const pct=Math.round(factor*100);$('zoomInfo').textContent=pct+'%';$('zoomPercent').value=pct;
 $('viewZoom').value=zoomState.mode==='manual'?(Array.from($('viewZoom').options).some(o=>o.value===String(zoomState.percent))?String(zoomState.percent):'custom'):zoomState.mode;
}
$('viewZoom').insertAdjacentHTML('beforeend','<option value="custom" hidden>自定义</option>');
$('viewZoom').addEventListener('change',()=>{const val=$('viewZoom').value;
 if(val==='fit'||val==='width')zoomState.mode=val;
 else if(val!=='custom'){zoomState.mode='manual';zoomState.percent=Number(val)}
 applyZoom();});
$('zoomPercent').addEventListener('change',()=>{zoomState.mode='manual';zoomState.percent=Math.max(10,Math.min(300,Number($('zoomPercent').value)||100));applyZoom()});
// Re-fit only when the viewport dimensions change; never during a paint stroke.
if(typeof ResizeObserver!=='undefined')new ResizeObserver(()=>{if(zoomState.mode!=='manual')applyZoom()}).observe($('canvasScroller'));
$('canvasScroller').addEventListener('wheel',e=>{
 if(!(e.ctrlKey||e.metaKey))return;e.preventDefault();
 const effective=Math.round((parseFloat(cvs.style.width)||cvs.width)/Math.max(cvs.width,1)*100);
 zoomState.mode='manual';zoomState.percent=Math.max(10,Math.min(300,effective+(e.deltaY<0?10:-10)));applyZoom();
},{passive:false});
for(const btn of document.querySelectorAll('[data-transfer-mode]'))btn.addEventListener('click',()=>{
 if(btn.dataset.transferMode==='precision'&&!nativeEngine.enabled){toast('精准蒙版需要本地原生 Python 引擎，请先启动 web_bridge.py');return;}
 state.transferMode=btn.dataset.transferMode;
 document.querySelectorAll('[data-transfer-mode]').forEach(b=>b.classList.toggle('active',b===btn));
 toast(state.transferMode==='precision'?'精准蒙版：直接使用桌面 Python 的 Precise Region 处理器':state.transferMode==='texture'?'纹理直贴：复杂背景从已配准的中文 SOURCE 迁入纹理和文字，仅作用于笔刷内':state.transferMode==='hole'?'挖孔替换：纯色气泡清底；复杂背景转为配准纹理移植':'透明底替换：纯色气泡清旧字；复杂背景使用柔边 SOURCE 纹理');
});

// Matching pages is not image registration. Every selected unedited pair needs a
// validated transform before displaying/migrating SOURCE. Jobs are per-pair and
// keyed by object identity so switching pages cannot overwrite a different pair.
const pendingAlignment=new WeakMap();
function isRegistered(p){return !!(p && Array.isArray(p.affine) && p.affine.length===6 && p.affine.every(Number.isFinite) && (!nativeEngine.enabled||Array.isArray(p.nativeMatrix)))}
function pairNeedsAlignment(p){return !!(p && p.src!==null && !p.skip && !isRegistered(p) && !p.edits.length && !p.alignError)}

function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toastHandle);toastHandle=setTimeout(()=>$('toast').classList.remove('show'),3900)}
function busy(on,title='',detail=''){ $('busy').classList.toggle('hidden',!on);if(on){$('busyTitle').textContent=title;$('busyDetail').textContent=detail} }
function current(){return state.pairs[state.index]||null}
function currentSrc(){const p=current();return p&&p.src!==null?state.sources[p.src]:null}
function currentTarget(){const p=current();return p?state.targets[p.tgt]:null}
function createPairs(){state.pairs=state.targets.map((_,i)=>defaultPair(i,state.sources.length));state.index=0;state.cloneAnchor=null;updateUI()}
function freshPairsForSources(){for(const p of state.pairs){if(!p.edits.length)p.src=p.tgt<state.sources.length?p.tgt:null;else if(p.src!==null&&p.src>=state.sources.length)p.src=null}updateUI()}
function setImportWiring(dropId,inputId,kind){
 const drop=$(dropId),input=$(inputId);drop.addEventListener('click',()=>input.click());drop.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();input.click()}});
 input.addEventListener('change',async()=>{const files=[...input.files];input.value='';if(files.length)await importImages(kind,files)});
 ['dragover','dragenter'].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.add('drag-over')}));
 ['dragleave','drop'].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.remove('drag-over')}));
 drop.addEventListener('drop',async e=>{const files=[...e.dataTransfer.files];if(files.length)await importImages(kind,files)});
}
setImportWiring('srcDrop','sourceInput','source');setImportWiring('tgtDrop','targetInput','target');
for(const [button,input,kind] of [['sourceFolder','sourceFolderInput','source'],['targetFolder','targetFolderInput','target']]){
 $(button).addEventListener('click',()=>$(input).click());$(input).addEventListener('change',async()=>{const f=[...$(input).files];$(input).value='';if(f.length)await importImages(kind,f)})
}
async function importImages(kind,files){
 if((state.targets.length||state.sources.length)&&state.pairs.some(p=>p.edits.length)&&!confirm('重新导入将清除尚未保存的编辑。请确认已保存工作区。'))return;
 busy(true,'读取图片',`解析 ${files.length} 个文件...`);
 try{
   const items=await unpack(files,kind);if(!items.length)throw Error('没有找到支持的图片（PNG / JPG / WebP / BMP / GIF）');
   if(items.length>1000)throw Error('单批最多导入 1000 页');
   const old=kind==='source'?state.sources:state.targets;
   releaseItems(old);
   if(kind==='source'){state.sources=items;state.pairs.forEach(p=>{p.edits=[];p.undo=[];p.redo=[];p.dx=0;p.dy=0;p.scale=1;p.affine=null;p.homography=null;p.nativeMatrix=null;p.alignDiagnostics=null;p.alignError=null});freshPairsForSources()}else{state.targets=items;createPairs()}
   toast(`已导入 ${items.length} 张${kind==='source'?'中文 SOURCE':'日文 TARGET'}图片`);
 }catch(err){toast(`导入失败：${err.message}`);console.error(err)}finally{busy(false);updateUI()}
}
function refreshPageList(){
 const list=$('pageList');list.replaceChildren();
 state.pairs.forEach((p,i)=>{
   const b=document.createElement('button');b.className='page-row'+(i===state.index?' active':'');b.type='button';b.dataset.index=i;
   const img=document.createElement('img');img.src=state.targets[p.tgt]?.url||'';img.loading='lazy';img.alt='';
   const meta=document.createElement('div');meta.className='page-meta';
   const title=document.createElement('strong');title.textContent=`${String(i+1).padStart(3,'0')}  ${state.targets[p.tgt]?.name||'未命名'}`;
   const sub=document.createElement('span');
   const alignLabel=p.src===null?'未配对':isRegistered(p)?'已配准':pendingAlignment.has(p)?'配准中':p.alignError?'配准失败':'待配准';
   sub.textContent=p.skip?'已跳过':`SOURCE ${p.src===null?'—':p.src+1} · ${alignLabel} · ${p.edits.length} 次操作`;
   meta.append(title,sub);const dot=document.createElement('span');dot.className='status-dot'+(p.skip?' skipped':(p.score!==null&&p.score<.24||p.edits.length?' good':''));dot.textContent=p.skip?'−':p.score!==null&&p.score>.3?'!':'●';
   b.append(img,meta,dot);b.addEventListener('click',()=>goPage(i));list.append(b);
 });
 if(!state.pairs.length){const d=document.createElement('div');d.className='empty-page';d.textContent='导入日文 TARGET 后显示页面';list.append(d)}
}
function updateUI(){
 $('sourceCount').textContent=state.sources.length+' 页';$('targetCount').textContent=state.targets.length+' 页';
 const p=current();$('pagePosition').textContent=p?`${state.index+1} / ${state.pairs.length}`:'— / —';
 $('previousPage').disabled=!p||state.index===0;$('nextPage').disabled=!p||state.index===state.pairs.length-1;
 $('exportAll').disabled=!p;$('exportPage').disabled=!p;$('saveProject').disabled=!p;
 $('autoPair').disabled=!state.sources.length||!state.targets.length;$('autoAlign').disabled=!currentSrc()||!currentTarget();
 $('currentPageTitle').textContent=p?`${String(state.index+1).padStart(3,'0')} · ${currentTarget()?.name||''}`:'尚未选择页面';
 $('sourceSelect').replaceChildren();const none=new Option('不配对中文页','');$('sourceSelect').add(none);state.sources.forEach((s,i)=>$('sourceSelect').add(new Option(`${String(i+1).padStart(3,'0')} · ${s.name}`,String(i))));
 $('sourceSelect').value=p&&p.src!==null?String(p.src):'';
 $('skipPage').checked=!!p?.skip;
 $('alignX').value=p?.dx||0;$('alignY').value=p?.dy||0;$('alignScale').value=100*(p?.scale||1);$('scaleValue').textContent=((p?.scale||1)*100).toFixed(1)+'%';
 $('alignStatus').textContent=isRegistered(p)?`已配准${p.homography?'（透视）':'（仿射）'} · 一致特征 ${p.alignDiagnostics?.inliers??'?'} 组 · 中位误差 ${(p.alignDiagnostics?.medianError??0).toFixed(2)} px`:
   pendingAlignment.has(p)?'正在自动配准当前页，请等待完成后再涂抹迁移':
   p?.alignError?`配准失败：${p.alignError}。请核对 SOURCE 页面、重试或手动调整`:
   p?.src===null?'尚未配对中文 SOURCE':'尚未对齐：系统将自动计算，未对齐前禁止文字迁移';
 $('editCount').textContent=(p?.edits.length||0)+' 次编辑';$('clearEdits').disabled=!p||!p.edits.length;$('undoButton').disabled=!p||!p.undo.length;$('redoButton').disabled=!p||!p.redo.length;
 $('pairSummary').textContent=state.targets.length?`${state.pairs.filter(x=>x.src!==null).length} / ${state.pairs.length} 页已配对 · ${state.pairs.filter(x=>x.skip).length} 页跳过`:'按自然文件名顺序配对；请手工核对。';
 refreshPageList();renderPage();
 // No need to press a separate button: automatically register the selected
 // page after import, pairing, page navigation or loading an old workspace.
 if(pairNeedsAlignment(p))void alignPage(p).catch(err=>console.error(err));
}
function goPage(index){if(index<0||index>=state.pairs.length)return;state.inspection=null;state.index=index;state.cloneAnchor=null;state.hover=null;updateUI()}
$('previousPage').addEventListener('click',()=>goPage(state.index-1));$('nextPage').addEventListener('click',()=>goPage(state.index+1));
$('sourceSelect').addEventListener('change',e=>{const p=current();if(!p)return;if(p.edits.length&&!confirm('修改 SOURCE 配对会改变本页已有迁移结果，确定继续？')){e.target.value=String(p.src??'');return}p.src=e.target.value===''?null:Number(e.target.value);p.affine=null;p.homography=null;p.nativeMatrix=null;p.alignDiagnostics=null;p.alignError=null;p.dx=0;p.dy=0;p.scale=1;updateUI()});
$('skipPage').addEventListener('change',e=>{const p=current();if(p){p.skip=e.target.checked;updateUI()}});
for(const name of ['alignX','alignY'])$(name).addEventListener('change',()=>{const p=current();if(!p)return;p.dx=Number($('alignX').value)||0;p.dy=Number($('alignY').value)||0;updateUI()});
$('alignScale').addEventListener('input',()=>{$('scaleValue').textContent=Number($('alignScale').value).toFixed(1)+'%'});
$('alignScale').addEventListener('change',()=>{const p=current();if(!p)return;p.scale=Number($('alignScale').value)/100;updateUI()});
function updateBrushSize(next){
 const size=Math.max(2,Math.min(240,Math.round(Number(next)||24)));
 state.brush=size;$('brushSize').value=size;$('brushSizeNumber').value=size;
 $('brushValue').textContent=size+' px';drawOverlay();
}
$('brushSize').addEventListener('input',()=>updateBrushSize($('brushSize').value));
$('brushSizeNumber').addEventListener('change',()=>updateBrushSize($('brushSizeNumber').value));
$('threshold').addEventListener('input',()=>{state.threshold=Number($('threshold').value);$('thresholdValue').textContent=state.threshold});
$('paperColor').addEventListener('input',()=>{state.paper=Number($('paperColor').value);$('paperValue').textContent=state.paper});
for(const button of document.querySelectorAll('[data-view]'))button.addEventListener('click',()=>{state.view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b===button));renderPage()});
for(const button of document.querySelectorAll('[data-tool]'))button.addEventListener('click',()=>{state.tool=button.dataset.tool;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b===button));drawOverlay()});
async function alignPage(pair,{manual=false}={}){
 if(!pair||pair.src===null||pair.skip)return false;
 if(pendingAlignment.has(pair))return pendingAlignment.get(pair);
 const sourceIndex=pair.src,targetIndex=pair.tgt;
 if(manual&&pair.edits.length&&!confirm('重新对齐会改变本页已有文字迁移效果。请先保存工作区，确定重新对齐？'))return false;
 if(!manual&&!pairNeedsAlignment(pair))return false;
 pair.alignError=null;
 const task=(async()=>{
   if(manual)busy(true,'正在重新配准','匹配 SOURCE 和 TARGET 结构特征...');
   else if(pair===current())$('alignStatus').textContent='正在自动配准，暂不可涂抹迁移...';
   try{
     const [src,tgt]=await Promise.all([bitmap(state.sources[sourceIndex]),bitmap(state.targets[targetIndex])]);
     const result=nativeEngine.enabled?
       await nativeRegister(state.sources[sourceIndex],state.targets[targetIndex]):
       await estimateBrowserAffine(src,tgt,detail=>{
       if(manual)$('busyDetail').textContent=detail;
     });
     // Never use an outdated result if the user imported/re-paired another page.
     if(!state.pairs.includes(pair)||pair.src!==sourceIndex||pair.tgt!==targetIndex)return false;
     pair.affine=nativeEngine.enabled?result.affine:result.matrix;
     pair.homography=nativeEngine.enabled?null:(result.homography||null);
     pair.nativeMatrix=nativeEngine.enabled?result.matrix:null;
     pair.alignDiagnostics={inliers:result.inliers,total:result.total,medianError:result.medianError,coverage:result.coverage,
       method:result.method||(result.homography?'browser-projective':'browser-affine'),confidence:result.confidence??null};
     pair.dx=0;pair.dy=0;pair.scale=1;pair.alignError=null;
     if(pair===current())toast(`本页已${result.homography?'透视':'仿射'}配准（${result.inliers} 组特征）；请在叠加检查中确认线条重合`);
     return true;
   }catch(e){
     if(state.pairs.includes(pair)&&pair.src===sourceIndex){
       pair.alignError=e.message||'不可靠的特征匹配';
       if(pair===current())toast(`自动配准失败：${pair.alignError}`);
     }
     return false;
   }finally{
     if(manual)busy(false);
   }
 })();
 pendingAlignment.set(pair,task);
 const ok=await task;
 pendingAlignment.delete(pair);
 if(state.pairs.includes(pair))updateUI();
 return ok;
}
$('autoAlign').addEventListener('click',()=>{
 const p=current();if(!p)return;
 if(pendingAlignment.has(p)){toast('当前页正在配准，请等待');return;}
 void alignPage(p,{manual:true});
});
$('resetAlign').addEventListener('click',()=>{
 const p=current();if(!p)return;if(p.edits.length&&!confirm('重置对齐会影响当前页的已有文字迁移，确定继续？'))return;
 p.affine=null;p.homography=null;p.nativeMatrix=null;p.alignDiagnostics=null;p.alignError=null;p.dx=0;p.dy=0;p.scale=1;updateUI();toast('已重置对齐');
});
$('autoPair').addEventListener('click',async()=>{
 if(!state.sources.length||!state.targets.length)return;
 if(state.pairs.some(p=>p.edits.length)&&!confirm('智能配对会改变页面对应关系，从而影响已编辑页面的迁移效果。是否继续？'))return;
 busy(true,'非 AI 页面结构配对','提取小尺寸结构特征...');
 try{const oldPairs=state.pairs;state.pairs=await smartPairs(state.sources,state.targets,state.pairs,text=>$('busyDetail').textContent=text);state.pairs.forEach((p,i)=>{if(p.src!==oldPairs[i]?.src){p.affine=null;p.homography=null;p.nativeMatrix=null;p.alignDiagnostics=null;p.alignError=null;p.dx=0;p.dy=0;p.scale=1}});toast('智能配对已完成；相似图片或封面仍可能配错，请核对后导出');updateUI()}
 catch(e){toast(`智能配对失败：${e.message}`);console.error(e)}finally{busy(false)}
});
// Independently inspect the SOURCE Chinese-ink and TARGET Japanese-ink
// candidates under an explicit user selection. No pixels are modified here.
function maskSelectionFor(pair){
 if(state.inspection?.pair===pair)return state.inspection.edit;
 return [...(pair?.edits||[])].reverse().find(e=>e.kind==='transfer')||null;
}
function inspectCurrentMask(target,source,p,edit){
 if(!source||!isRegistered(p))throw Error('请先配对并自动配准这一页');
 if(!edit)throw Error('先用「◉ 差异检查」涂抹需要分析的区域');
 const box=getBounds(edit,target.width,target.height);
 if(!box||box.w*box.h>4_000_000)throw Error('选区太大，请分成较小区域检查（上限400万像素）');
 const sourceWarp=warpedSource(source,target.width,target.height,p);
 const sc=sourceWarp.getContext('2d',{willReadFrequently:true});
 const tc=document.createElement('canvas');tc.width=box.w;tc.height=box.h;
 const tctx=tc.getContext('2d',{willReadFrequently:true});
 tctx.drawImage(target,-box.x,-box.y);
 const sourcePixels=sc.getImageData(box.x,box.y,box.w,box.h).data;
 const targetPixels=tctx.getImageData(0,0,box.w,box.h).data;
 const selection=brushOrSelectionMask(edit,box.x,box.y,box.w,box.h);
 const inspection=inspectTextDifferences(sourcePixels,targetPixels,selection,box.w,box.h);
 tc.width=tc.height=0;
 return {box,inspection};
}
function oneMaskPng(mask,w,h){
 const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;
 const context=canvas.getContext('2d');const image=context.createImageData(w,h);
 for(let i=0;i<mask.length;i++){let j=i*4;const value=mask[i]?255:0;image.data[j]=image.data[j+1]=image.data[j+2]=value;image.data[j+3]=255}
 context.putImageData(image,0,0);return canvas;
}
async function renderPage(){
 const stamp=++state.stamp;const p=current();if(!p){$('emptyState').classList.remove('hidden');$('canvasScroller').classList.add('hidden');$('imageDims').textContent='未加载图片';return}
 try{
   const target=await bitmap(currentTarget());const source=currentSrc()?await bitmap(currentSrc()):null;
   if(stamp!==state.stamp)return;
   const W=target.width,H=target.height;
   const changedPage=zoomState.pageId!==currentTarget()?.id;
   if(cvs.width!==W)cvs.width=W;if(cvs.height!==H)cvs.height=H;
   if(over.width!==W)over.width=W;if(over.height!==H)over.height=H;
   ctx.clearRect(0,0,W,H);
   if(state.view==='result' && nativeEngine.enabled && p.edits.length){
     const response=await nativeRender(p,currentSrc(),currentTarget());
     if(stamp!==state.stamp)return;
     const img=await decodedPNG(response.png);if(stamp!==state.stamp)return;
     ctx.drawImage(img,0,0);state.pendingStroke=null;
     const tail=response.receipts.at(-1);
     $('renderHint').textContent='原生 Python 处理 · '+(tail?tail.mode+' · '+(tail.changed??0)+' 处像素变更':'已处理')+
       '；预览与导出调用同一原生区域处理函数；画笔越界写入由桌面源码校验';
   }else if(state.view==='result'){const guarded=[],strokeResults=[];const composite=compose(target,source,p,{threshold:state.threshold,paper:state.paper,onGuard:(reason,edit)=>guarded.push({reason,edit}),onStrokeResult:(edit,result)=>strokeResults.push({edit,result})});
     if(guarded.length){$('renderHint').textContent='⚠ '+guarded.at(-1).reason+'。请检查配准或缩小画笔。';}
     else $('renderHint').textContent='纯色区域贴字；建筑／网点等复杂背景自动采用 SOURCE 配准纹理，不会涂白；纹理色差需要人工检查';
     ctx.drawImage(composite,0,0);composite.width=0;composite.height=0;
     const newest=state.pendingStroke;
     if(newest&&newest.pair===p&&p.edits.at(-1)===newest.edit){
       state.pendingStroke=null;
       const failed=guarded.find(x=>x.edit===newest.edit);
       const stats=strokeResults.find(x=>x.edit===newest.edit)?.result;
       if(!failed&&stats?.method==='registered-texture'){
         $('renderHint').textContent='已使用 SOURCE 配准纹理迁移：未清白背景；请检查画笔边缘是否有扫描色差，必要时右键恢复';
       }
       if(failed||!stats?.applied||stats.changed===0){
         // Unusable strokes must not enter undo/export history as if valid.
         p.edits.pop();p.undo.pop();p.redo=[];
         $('editCount').textContent=p.edits.length+' 次编辑';
         $('clearEdits').disabled=!p.edits.length;
         $('undoButton').disabled=!p.undo.length;$('redoButton').disabled=true;
         toast('本次未替换：'+(failed?.reason||'刷过的区域没有可迁移的不同文字像素')+'。可以调整笔刷位置或尺寸。');
         queueMicrotask(()=>{if(p===current())renderPage()});
       }
     }
   }
   else if(state.view==='masks'){
     ctx.drawImage(target,0,0);
     try{
       const {box,inspection}=inspectCurrentMask(target,source,p,maskSelectionFor(p));
       const maskCanvas=document.createElement('canvas');maskCanvas.width=box.w;maskCanvas.height=box.h;
       const mc=maskCanvas.getContext('2d');const image=mc.createImageData(box.w,box.h);
       image.data.set(maskOverlay(inspection,box.w,box.h));mc.putImageData(image,0,0);
       ctx.drawImage(maskCanvas,box.x,box.y);maskCanvas.width=maskCanvas.height=0;
       const x=inspection.summary;
       $('renderHint').textContent=`只读掩码 · 绿 SOURCE 中文候选 ${x.sourceCandidates}px · 红 TARGET 日文候选 ${x.targetCandidates}px · 公共结构 ${Math.round(x.structuralSupport*100)}% · ⚠ ${x.caution}`;
     }catch(error){$('renderHint').textContent='掩码尚未生成：'+error.message;}
   }
   else if(state.view==='target')ctx.drawImage(target,0,0);
   else if(state.view==='overlay'||state.view==='source'){
     if(state.view==='overlay')ctx.drawImage(target,0,0);else{ctx.fillStyle='white';ctx.fillRect(0,0,W,H)}
     if(source){
       if(nativeEngine.enabled && Array.isArray(p.nativeMatrix)){
         const warped=await nativeWarp(p,currentSrc(),currentTarget());if(stamp!==state.stamp)return;
         const img=await decodedPNG(warped.png);if(stamp!==state.stamp)return;
         ctx.save();if(state.view==='overlay')ctx.globalAlpha=.48;ctx.drawImage(img,0,0);ctx.restore();
       }else{
         const warped=warpedSource(source,W,H,p);
         ctx.save();if(state.view==='overlay')ctx.globalAlpha=.48;ctx.drawImage(warped,0,0);ctx.restore();warped.width=0;warped.height=0;
       }
     }else if(state.view==='source')ctx.drawImage(target,0,0);
   }
   if(stamp!==state.stamp)return;
   $('emptyState').classList.add('hidden');$('canvasScroller').classList.remove('hidden');
   zoomState.pageId=currentTarget()?.id;
   // Pixel-only redraw after a stroke: never touch CSS size or scroll offsets.
   if(changedPage||!cvs.style.width)applyZoom({resetCenter:changedPage});
   $('imageDims').textContent=`${W} × ${H} px · ${p.edits.length} 次编辑`;
   drawOverlay();
 }catch(err){
   if(stamp===state.stamp){
     toast(`加载页面失败：${err.message}`);
     if(nativeEngine.enabled&&state.pendingStroke?.pair===p&&p.edits.at(-1)===state.pendingStroke.edit){
       p.edits.pop();p.undo.pop();p.redo=[];state.pendingStroke=null;
       $('editCount').textContent=p.edits.length+' 次编辑';
       queueMicrotask(()=>{if(p===current())renderPage()});
     }
   }
   console.error(err)
 }
}
function pointFromEvent(e){const r=over.getBoundingClientRect();return{x:Math.round((e.clientX-r.left)*over.width/r.width),y:Math.round((e.clientY-r.top)*over.height/r.height)}}
function drawOverlay(){
 oc.clearRect(0,0,over.width,over.height);
 const e=state.gesture;
 if(!e){
  const pointer=state.hover;
  if(pointer && ['paint-transfer','erase','clone','restore','mask-inspect'].includes(state.tool)){
   oc.save();oc.setLineDash([5,4]);oc.strokeStyle='#21c9bd';oc.fillStyle='#00dfca1b';
   oc.lineWidth=Math.max(2,over.width/700);
   oc.beginPath();oc.arc(pointer.x,pointer.y,state.brush/2,0,Math.PI*2);oc.fill();oc.stroke();oc.restore();
  }
  return;
 }
 oc.lineWidth=Math.max(2,over.width/450);oc.strokeStyle='#00dfca';oc.fillStyle='#00dfca24';oc.setLineDash([8,6]);
 if(e.shape==='rect'){const x=Math.min(e.start.x,e.end.x),y=Math.min(e.start.y,e.end.y),w=Math.abs(e.start.x-e.end.x),h=Math.abs(e.start.y-e.end.y);oc.fillRect(x,y,w,h);oc.strokeRect(x,y,w,h)}
 else if(e.shape==='ellipse'){const x=(e.start.x+e.end.x)/2,y=(e.start.y+e.end.y)/2;oc.beginPath();oc.ellipse(x,y,Math.max(1,Math.abs(e.start.x-e.end.x)/2),Math.max(1,Math.abs(e.start.y-e.end.y)/2),0,0,Math.PI*2);oc.fill();oc.stroke()}
 else {oc.setLineDash([]);oc.lineWidth=e.brush;oc.lineCap='round';oc.strokeStyle='#00dfca85';oc.beginPath();const pts=e.points||[];if(pts.length){oc.moveTo(pts[0].x,pts[0].y);pts.forEach(p=>oc.lineTo(p.x,p.y));oc.stroke();if(pts.length===1){oc.beginPath();oc.arc(pts[0].x,pts[0].y,e.brush/2,0,2*Math.PI);oc.fill()}}}
}
over.addEventListener('contextmenu',e=>e.preventDefault());
over.addEventListener('pointerdown',e=>{
 if(!currentTarget())return;if(e.button!==0&&e.button!==2)return;
 const rightRestore=e.button===2;
 const point=pointFromEvent(e);if(!rightRestore&&state.tool==='clone'&&e.altKey){state.cloneAnchor=point;toast(`克隆取样点 (${point.x}, ${point.y}) 已设定`);return}
 if(!rightRestore&&state.tool==='clone'&&!state.cloneAnchor){toast('请先按住 Alt，在原图的干净背景处单击取样');return}
 if(!rightRestore&&['ellipse','rect','paint-transfer','mask-inspect'].includes(state.tool)&&!currentSrc()){toast('当前页面尚未配对 SOURCE，无法迁移中文文字');return}
 if(!rightRestore&&['ellipse','rect','paint-transfer','mask-inspect'].includes(state.tool)&&!isRegistered(current())){
   const p=current();toast(p.alignError?'配准失败，请检查对应页或使用桌面版':'正在自动配准当前页，完成后才能涂抹迁移');
   if(pairNeedsAlignment(p))void alignPage(p);
   return;
 }
 if(state.view!=='result'){state.view='result';document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view==='result'));/* Deferred redraw until stroke completes to retain pointer capture. */}
 const tool=rightRestore?'restore':state.tool;
 const shape=['ellipse','rect'].includes(tool)?tool:'brush';
 state.gesture={kind:['ellipse','rect','paint-transfer'].includes(tool)?'transfer':tool,
   mode:state.transferMode,shape,start:point,end:point,points:shape==='brush'?[point]:undefined,
   brush:state.brush,threshold:state.threshold,paper:state.paper};
 if(tool==='clone')state.gesture.cloneOffset={x:state.cloneAnchor.x-point.x,y:state.cloneAnchor.y-point.y};
 try{over.setPointerCapture(e.pointerId)}catch{}drawOverlay();e.preventDefault();
});
over.addEventListener('pointermove',e=>{const p=pointFromEvent(e);state.hover=p;if(!state.gesture){drawOverlay();return}const g=state.gesture;g.end=p;if(g.shape==='brush'){const prev=g.points.at(-1),d=Math.hypot(prev.x-p.x,prev.y-p.y);if(d>=1.7)g.points.push(p)}drawOverlay()});
function finishGesture(e){const g=state.gesture;if(!g)return;state.gesture=null;drawOverlay();
 if(state.tool==='mask-inspect'&&g.kind==='mask-inspect'){
   const p=current();if(!p)return;
   state.inspection={pair:p,edit:g};state.view='masks';
   document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view==='masks'));
   renderPage();toast('差异掩码：绿色候选中文、红色候选日文；只是诊断，不会修改或清除漫画');return;
 }
 if(g.shape!=='brush'&&(Math.abs(g.end.x-g.start.x)<10||Math.abs(g.end.y-g.start.y)<10)){toast('选区太小，请拖动较大的气泡内部区域');return}
 const p=current();if(!p)return;p.undo.push(structuredClone(p.edits));if(p.undo.length>50)p.undo.shift();p.redo=[];p.edits.push(g);state.pendingStroke=g.kind==='transfer'?{pair:p,edit:g}:null;
 // Keep the zoom, current scroll offset, mouse position and page list stable.
 $('editCount').textContent=p.edits.length+' 次编辑';$('clearEdits').disabled=!p.edits.length;
 $('undoButton').disabled=!p.undo.length;$('redoButton').disabled=true;
 renderPage();}
over.addEventListener('pointerup',finishGesture);over.addEventListener('pointercancel',()=>{state.gesture=null;drawOverlay()});over.addEventListener('pointerleave',()=>{if(!state.gesture){state.hover=null;drawOverlay()}});
function undo(){const p=current();if(!p||!p.undo.length)return;p.redo.push(structuredClone(p.edits));p.edits=p.undo.pop();updateUI()}
function redo(){const p=current();if(!p||!p.redo.length)return;p.undo.push(structuredClone(p.edits));p.edits=p.redo.pop();updateUI()}
$('undoButton').addEventListener('click',undo);$('redoButton').addEventListener('click',redo);
$('clearEdits').addEventListener('click',()=>{const p=current();if(!p||!p.edits.length)return;if(!confirm('清空本页全部编辑？可以用撤销恢复。'))return;p.undo.push(structuredClone(p.edits));p.edits=[];p.redo=[];updateUI()});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?redo():undo()}if(e.key==='ArrowRight')goPage(state.index+1);if(e.key==='ArrowLeft')goPage(state.index-1);if(e.key==='['||e.key===']'){e.preventDefault();updateBrushSize(state.brush+(e.key===']'?8:-8))}});
async function exportOne(){const p=current();if(!p)return;
 if(p.edits.some(e=>e.kind==='transfer')&&!isRegistered(p)){toast('本页含未配准的迁移操作，禁止错位导出；请先自动配准');return;}
 busy(true,'生成 PNG','按原始 TARGET 分辨率合成...');try{
   let blob;
   if(nativeEngine.enabled && p.edits.length){
     const r=await nativeRender(p,currentSrc(),currentTarget());blob=await(await fetch(r.png)).blob();
   }else{
     const [t,s]=await Promise.all([bitmap(currentTarget()),currentSrc()?bitmap(currentSrc()):null]);
     const c=compose(t,s,p,{rejectUnsafe:true});blob=await canvasBlob(c);c.width=0;c.height=0;
   }
   download(blob,safeName(currentTarget().name).replace(/\.[^.]+$/,'')+'_folirina.png');toast('本页 PNG 已生成')
 }catch(e){toast(`导出失败：${e.message}`);console.error(e)}finally{busy(false)}}
$('exportPage').addEventListener('click',exportOne);
$('exportAll').addEventListener('click',async()=>{
 if(!state.targets.length)return;busy(true,'导出整册','正在生成 PNG 文件...');try{
 const zip=new JSZip(),dir=zip.folder('Folirina_Output');let count=0;
 for(let i=0;i<state.pairs.length;i++){
  const p=state.pairs[i];if(p.skip)continue;
  if(p.edits.some(e=>e.kind==='transfer')&&!isRegistered(p))throw Error(`第 ${i+1} 页包含未配准文字迁移，请先自动配准，避免错位导出`);
  $('busyDetail').textContent=`合成页面 ${i+1}/${state.pairs.length}`;
  const t=await bitmap(state.targets[p.tgt]),s=p.src===null?null:await bitmap(state.sources[p.src]);
  let blob;
  if(nativeEngine.enabled&&p.edits.length){const r=await nativeRender(p,state.sources[p.src],state.targets[p.tgt]);blob=await(await fetch(r.png)).blob();}
  else{const c=compose(t,s,p,{rejectUnsafe:true});blob=await canvasBlob(c);c.width=0;c.height=0;}
  dir.file(`${String(i+1).padStart(4,'0')}_${safeName(state.targets[p.tgt].name).replace(/\.[^.]+$/,'')}.png`,blob);count++;
  await new Promise(r=>setTimeout(r,0));
 }
 if(!count)throw Error('所有页面均被跳过');
 $('busyDetail').textContent='压缩 ZIP...';const out=await zip.generateAsync({type:'blob',compression:'STORE'},meta=>{$('busyDetail').textContent=`压缩 ZIP ${Math.round(meta.percent)}%`});download(out,'Folirina_Web_Export.zip');toast(`已导出 ${count} 页 PNG，按 TARGET 原分辨率保存`);
 }catch(e){toast(`导出失败：${e.message}`);console.error(e)}finally{busy(false)}
});
$('exportMasks').addEventListener('click',async()=>{
 const p=current();if(!p)return;
 busy(true,'导出差异掩码','分别提取 SOURCE / TARGET 的候选字形...');
 try{
  const [target,source]=await Promise.all([bitmap(currentTarget()),currentSrc()?bitmap(currentSrc()):null]);
  const {box,inspection}=inspectCurrentMask(target,source,p,maskSelectionFor(p));
  const zip=new JSZip();const files=[['source_candidates.png',inspection.sourceMask],['target_candidates.png',inspection.targetMask]];
  for(const [name,mask] of files){const image=oneMaskPng(mask,box.w,box.h);const blob=await canvasBlob(image);zip.file(name,blob);image.width=image.height=0}
  zip.file('diagnostic.json',JSON.stringify({page:currentTarget().name,bounds:box,summary:inspection.summary,warning:'仅候选区域，非可靠的文字识别；不得未经检查直接擦除'},null,2));
  download(await zip.generateAsync({type:'blob',compression:'DEFLATE'}),safeName(currentTarget().name)+'_dual_masks.zip');
  toast('已导出 SOURCE／TARGET 独立候选掩码，仅供人工对照');
 }catch(err){toast('掩码导出失败：'+err.message)}finally{busy(false)}
});
$('saveProject').addEventListener('click',async()=>{
 if(!state.targets.length)return;busy(true,'打包工作区','保存原图、配对关系、编辑记录...');
 try{const zip=new JSZip();
 const sources=state.sources.map((i,k)=>({name:i.name,path:`source/${String(k).padStart(4,'0')}.${(i.name.split('.').at(-1)||'png').toLowerCase()}`}));
 const targets=state.targets.map((i,k)=>({name:i.name,path:`target/${String(k).padStart(4,'0')}.${(i.name.split('.').at(-1)||'png').toLowerCase()}`}));
 state.sources.forEach((i,k)=>zip.file(sources[k].path,i.blob));state.targets.forEach((i,k)=>zip.file(targets[k].path,i.blob));
 zip.file('project.json',JSON.stringify({format:'folirina-web-lite',version:1,baseCommit:'b87edcd124a16cdd9feec7b42934291c53c16bbc',sources,targets,pairs:state.pairs.map(({undo,redo,...p})=>p),index:state.index},null,2));
 const out=await zip.generateAsync({type:'blob',compression:'STORE'},m=>$('busyDetail').textContent=`打包 ${Math.round(m.percent)}%`);download(out,'Folirina_Web_Workspace.zip');toast('工作区已保存；下次可直接载入 ZIP 继续编辑');
 }catch(e){toast(`保存失败：${e.message}`);console.error(e)}finally{busy(false)}
});
$('openProject').addEventListener('click',()=>$('projectInput').click());
$('projectInput').addEventListener('change',async e=>{
 const f=e.target.files?.[0];e.target.value='';if(!f)return;
 if(state.pairs.some(p=>p.edits.length)&&!confirm('打开工作区将替换当前未保存的编辑，是否继续？'))return;
 busy(true,'加载工作区','读取 project.json 和原始图像...');let newSources=[],newTargets=[];
 try{
 const zip=await JSZip.loadAsync(f),mf=zip.file('project.json');if(!mf)throw Error('找不到 project.json，不是 Folirina 工作区');
 const data=JSON.parse(await mf.async('string'));if(data.format!=='folirina-web-lite'||data.version!==1)throw Error('工作区格式或版本不受支持');
 if(!Array.isArray(data.sources)||!Array.isArray(data.targets)||!Array.isArray(data.pairs)||data.targets.length!==data.pairs.length||data.targets.length>1000||data.sources.length>1000)throw Error('工作区元数据不完整');
 for(const [kind,descs] of [['source',data.sources],['target',data.targets]]){
  const items=[];for(let i=0;i<descs.length;i++){const d=descs[i],entry=zip.file(d.path);if(!entry||!/^(source|target)\/\d{4,}\.(png|jpe?g|webp|bmp|gif)$/i.test(d.path))throw Error('工作区图像缺失或路径非法');const blob=await entry.async('blob');items.push({id:`workspace-${kind}-${i}-${Date.now()}`,name:d.name,blob,url:URL.createObjectURL(blob)})}
  if(kind==='source')newSources=items;else newTargets=items;
 }
 const pairs=data.pairs.map((p,i)=>{if((p.homography!=null&&(!Array.isArray(p.homography)||p.homography.length!==9||p.homography.some(n=>typeof n!=='number'||!Number.isFinite(n)||Math.abs(n)>100000)))||p.tgt!==i||!(p.src===null||Number.isInteger(p.src)&&p.src>=0&&p.src<newSources.length)||!Array.isArray(p.edits)||p.edits.length>5000||p.edits.some(e=>!'transfer erase clone restore'.split(' ').includes(e.kind)||!'rect ellipse brush'.split(' ').includes(e.shape))||(p.affine!==null&&p.affine!==undefined&&(!Array.isArray(p.affine)||p.affine.length!==6||p.affine.some(n=>typeof n!=='number'||!Number.isFinite(n)||Math.abs(n)>100000))))throw Error('配对数据错误');return {...defaultPair(i,newSources.length),...p,undo:[],redo:[]}});
 releaseItems(state.sources);releaseItems(state.targets);state.sources=newSources;state.targets=newTargets;state.pairs=pairs;state.index=Math.min(Math.max(0,data.index||0),pairs.length-1);newSources=[];newTargets=[];updateUI();toast(`工作区加载成功，共 ${pairs.length} 页`);
 }catch(err){releaseItems(newSources);releaseItems(newTargets);toast(`加载失败：${err.message}`);console.error(err)}finally{busy(false)}
});
if('serviceWorker' in navigator && location.protocol.startsWith('http'))navigator.serviceWorker.register('./sw.js').catch(()=>{});
updateUI();
// For smoke/regression tests only: no privileged APIs or network actions.
window.__folirinaTest={state,nativeEngine,nativeRender,nativeWarp,composeCurrent:async()=>{const p=current();return compose(await bitmap(currentTarget()),currentSrc()?await bitmap(currentSrc()):null,p)},goPage};
