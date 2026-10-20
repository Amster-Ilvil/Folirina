import {unpack,releaseItems,bitmap,defaultPair,compose,smartPairs,canvasBlob,download,naturalCompare,safeName,warpedSource} from './core.js';
import {estimateBrowserAffine} from './registration.js';

const $=id=>document.getElementById(id);
const state={sources:[],targets:[],pairs:[],index:0,view:'result',tool:'ellipse',brush:24,threshold:185,paper:255,gesture:null,cloneAnchor:null,stamp:0,rendering:false};
const cvs=$('editorCanvas'),over=$('overlayCanvas'),ctx=cvs.getContext('2d',{willReadFrequently:true}),oc=over.getContext('2d');
let toastHandle=null;
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
   if(kind==='source'){state.sources=items;state.pairs.forEach(p=>{p.edits=[];p.undo=[];p.redo=[];p.dx=0;p.dy=0;p.scale=1;p.affine=null;p.alignDiagnostics=null});freshPairsForSources()}else{state.targets=items;createPairs()}
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
   const sub=document.createElement('span');sub.textContent=p.skip?'已跳过':`${p.src===null?'未配对':`SOURCE ${p.src+1}`} · ${p.edits.length} 次操作`;
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
 $('alignStatus').textContent=p?.affine?`有效匹配 ${p.alignDiagnostics?.inliers??'?'} 组 · 缩略图中位误差 ${(p.alignDiagnostics?.medianError||0).toFixed(2)} px`:'尚未自动配准：旧版仅缩放和平移';
 $('editCount').textContent=(p?.edits.length||0)+' 次编辑';$('clearEdits').disabled=!p||!p.edits.length;$('undoButton').disabled=!p||!p.undo.length;$('redoButton').disabled=!p||!p.redo.length;
 $('pairSummary').textContent=state.targets.length?`${state.pairs.filter(x=>x.src!==null).length} / ${state.pairs.length} 页已配对 · ${state.pairs.filter(x=>x.skip).length} 页跳过`:'按自然文件名顺序配对；请手工核对。';
 refreshPageList();renderPage();
}
function goPage(index){if(index<0||index>=state.pairs.length)return;state.index=index;state.cloneAnchor=null;updateUI()}
$('previousPage').addEventListener('click',()=>goPage(state.index-1));$('nextPage').addEventListener('click',()=>goPage(state.index+1));
$('sourceSelect').addEventListener('change',e=>{const p=current();if(!p)return;if(p.edits.length&&!confirm('修改 SOURCE 配对会改变本页已有迁移结果，确定继续？')){e.target.value=String(p.src??'');return}p.src=e.target.value===''?null:Number(e.target.value);p.affine=null;p.alignDiagnostics=null;p.dx=0;p.dy=0;p.scale=1;updateUI()});
$('skipPage').addEventListener('change',e=>{const p=current();if(p){p.skip=e.target.checked;updateUI()}});
for(const name of ['alignX','alignY'])$(name).addEventListener('change',()=>{const p=current();if(!p)return;p.dx=Number($('alignX').value)||0;p.dy=Number($('alignY').value)||0;updateUI()});
$('alignScale').addEventListener('input',()=>{$('scaleValue').textContent=Number($('alignScale').value).toFixed(1)+'%'});
$('alignScale').addEventListener('change',()=>{const p=current();if(!p)return;p.scale=Number($('alignScale').value)/100;updateUI()});
for(const type of ['threshold','brushSize','paperColor'])$(type).addEventListener('input',()=>{
 state.threshold=Number($('threshold').value);state.brush=Number($('brushSize').value);state.paper=Number($('paperColor').value);
 $('thresholdValue').textContent=String(state.threshold);$('brushValue').textContent=state.brush+' px';$('paperValue').textContent=String(state.paper);
});
for(const button of document.querySelectorAll('[data-view]'))button.addEventListener('click',()=>{state.view=button.dataset.view;document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b===button));renderPage()});
for(const button of document.querySelectorAll('[data-tool]'))button.addEventListener('click',()=>{state.tool=button.dataset.tool;document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b===button));drawOverlay()});
$('autoAlign').addEventListener('click',async()=>{
 const p=current();if(!p||p.src===null)return;
 if(p.edits.length&&!confirm('重新对齐会改变本页已有文字迁移效果。建议先保存工作区，是否继续？'))return;
 const targetIndex=p.tgt,sourceIndex=p.src;
 busy(true,'非 AI 画面配准','检测特征并计算仿射变换（无需模型）...');
 try{
  const [src,tgt]=await Promise.all([bitmap(state.sources[sourceIndex]),bitmap(state.targets[targetIndex])]);
  const result=await estimateBrowserAffine(src,tgt,text=>$('busyDetail').textContent=text);
  const active=state.pairs.find(q=>q.tgt===targetIndex);
  if(active&&active.src===sourceIndex){
   active.affine=result.matrix;active.alignDiagnostics={inliers:result.inliers,total:result.total,medianError:result.medianError,coverage:result.coverage};
   active.dx=0;active.dy=0;active.scale=1;
   toast(`自动配准成功：${result.inliers} 组一致特征点。请打开「叠加检查」核对画框和气泡！`);updateUI();
  }
 }catch(e){toast(`自动配准失败，旧对齐已保留：${e.message}`);console.error(e)}finally{busy(false)}
});
$('resetAlign').addEventListener('click',()=>{
 const p=current();if(!p)return;if(p.edits.length&&!confirm('重置对齐会影响当前页的已有文字迁移，确定继续？'))return;
 p.affine=null;p.alignDiagnostics=null;p.dx=0;p.dy=0;p.scale=1;updateUI();toast('已重置对齐');
});
$('autoPair').addEventListener('click',async()=>{
 if(!state.sources.length||!state.targets.length)return;
 if(state.pairs.some(p=>p.edits.length)&&!confirm('智能配对会改变页面对应关系，从而影响已编辑页面的迁移效果。是否继续？'))return;
 busy(true,'非 AI 页面结构配对','提取小尺寸结构特征...');
 try{const oldPairs=state.pairs;state.pairs=await smartPairs(state.sources,state.targets,state.pairs,text=>$('busyDetail').textContent=text);state.pairs.forEach((p,i)=>{if(p.src!==oldPairs[i]?.src){p.affine=null;p.alignDiagnostics=null;p.dx=0;p.dy=0;p.scale=1}});toast('智能配对已完成；相似图片或封面仍可能配错，请核对后导出');updateUI()}
 catch(e){toast(`智能配对失败：${e.message}`);console.error(e)}finally{busy(false)}
});
async function renderPage(){
 const stamp=++state.stamp;const p=current();if(!p){$('emptyState').classList.remove('hidden');$('canvasScroller').classList.add('hidden');$('imageDims').textContent='未加载图片';return}
 try{
   const target=await bitmap(currentTarget());const source=currentSrc()?await bitmap(currentSrc()):null;
   if(stamp!==state.stamp)return;
   const W=target.width,H=target.height;
   cvs.width=W;cvs.height=H;over.width=W;over.height=H;
   ctx.clearRect(0,0,W,H);
   if(state.view==='result'){const composite=compose(target,source,p,{threshold:state.threshold,paper:state.paper});ctx.drawImage(composite,0,0);composite.width=0;composite.height=0}
   else if(state.view==='target')ctx.drawImage(target,0,0);
   else if(state.view==='overlay'){ctx.drawImage(target,0,0);if(source){ctx.save();ctx.globalAlpha=.48;const warped=warpedSource(source,W,H,p);ctx.drawImage(warped,0,0);ctx.restore();warped.width=0;warped.height=0;}}
   else {ctx.fillStyle='white';ctx.fillRect(0,0,W,H);if(source)ctx.drawImage(warpedSource(source,W,H,p),0,0);else ctx.drawImage(target,0,0)}
   if(stamp!==state.stamp)return;
   $('emptyState').classList.add('hidden');$('canvasScroller').classList.remove('hidden');
   const rect=$('canvasScroller').getBoundingClientRect(),availableW=Math.max(200,rect.width-60),availableH=Math.max(190,rect.height-55),factor=Math.min(1.35,availableW/W,availableH/H);
   const width=Math.max(10,Math.round(W*factor)),height=Math.max(10,Math.round(H*factor));
   cvs.style.width=over.style.width=$('canvasHolder').style.width=width+'px';cvs.style.height=over.style.height=$('canvasHolder').style.height=height+'px';
   $('emptyState').classList.add('hidden');$('canvasScroller').classList.remove('hidden');$('imageDims').textContent=`${W} × ${H} px · ${p.edits.length} 次编辑`;$('zoomInfo').textContent=`${Math.round(factor*100)}%`;
   drawOverlay();
 }catch(err){if(stamp===state.stamp)toast(`加载页面失败：${err.message}`);console.error(err)}
}
function pointFromEvent(e){const r=over.getBoundingClientRect();return{x:Math.round((e.clientX-r.left)*over.width/r.width),y:Math.round((e.clientY-r.top)*over.height/r.height)}}
function drawOverlay(){
 oc.clearRect(0,0,over.width,over.height);
 const e=state.gesture;if(!e)return;
 oc.lineWidth=Math.max(2,over.width/450);oc.strokeStyle='#00dfca';oc.fillStyle='#00dfca24';oc.setLineDash([8,6]);
 if(e.shape==='rect'){const x=Math.min(e.start.x,e.end.x),y=Math.min(e.start.y,e.end.y),w=Math.abs(e.start.x-e.end.x),h=Math.abs(e.start.y-e.end.y);oc.fillRect(x,y,w,h);oc.strokeRect(x,y,w,h)}
 else if(e.shape==='ellipse'){const x=(e.start.x+e.end.x)/2,y=(e.start.y+e.end.y)/2;oc.beginPath();oc.ellipse(x,y,Math.max(1,Math.abs(e.start.x-e.end.x)/2),Math.max(1,Math.abs(e.start.y-e.end.y)/2),0,0,Math.PI*2);oc.fill();oc.stroke()}
 else {oc.setLineDash([]);oc.lineWidth=e.brush;oc.lineCap='round';oc.strokeStyle='#00dfca85';oc.beginPath();const pts=e.points||[];if(pts.length){oc.moveTo(pts[0].x,pts[0].y);pts.forEach(p=>oc.lineTo(p.x,p.y));oc.stroke();if(pts.length===1){oc.beginPath();oc.arc(pts[0].x,pts[0].y,e.brush/2,0,2*Math.PI);oc.fill()}}}
}
over.addEventListener('pointerdown',e=>{
 if(!currentTarget())return;if(e.button!==0)return;
 const point=pointFromEvent(e);if(state.tool==='clone'&&e.altKey){state.cloneAnchor=point;toast(`克隆取样点 (${point.x}, ${point.y}) 已设定`);return}
 if(state.tool==='clone'&&!state.cloneAnchor){toast('请先按住 Alt，在原图的干净背景处单击取样');return}
 if(['ellipse','rect'].includes(state.tool)&&!currentSrc()){toast('当前页面尚未配对 SOURCE，无法迁移中文文字');return}
 if(state.view!=='result'){state.view='result';document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view==='result'));renderPage()}
 const shape=['ellipse','rect'].includes(state.tool)?state.tool:'brush';state.gesture={kind:['ellipse','rect'].includes(state.tool)?'transfer':state.tool,shape,start:point,end:point,points:shape==='brush'?[point]:undefined,brush:state.brush,threshold:state.threshold,paper:state.paper};
 if(state.tool==='clone')state.gesture.cloneOffset={x:state.cloneAnchor.x-point.x,y:state.cloneAnchor.y-point.y};
 try{over.setPointerCapture(e.pointerId)}catch{}drawOverlay();e.preventDefault();
});
over.addEventListener('pointermove',e=>{if(!state.gesture)return;const p=pointFromEvent(e),g=state.gesture;g.end=p;if(g.shape==='brush'){const prev=g.points.at(-1),d=Math.hypot(prev.x-p.x,prev.y-p.y);if(d>=1.7)g.points.push(p)}drawOverlay()});
function finishGesture(e){const g=state.gesture;if(!g)return;state.gesture=null;drawOverlay();if(g.shape!=='brush'&&(Math.abs(g.end.x-g.start.x)<10||Math.abs(g.end.y-g.start.y)<10)){toast('选区太小，请拖动较大的气泡内部区域');return}
 const p=current();if(!p)return;p.undo.push(structuredClone(p.edits));if(p.undo.length>50)p.undo.shift();p.redo=[];p.edits.push(g);updateUI();}
over.addEventListener('pointerup',finishGesture);over.addEventListener('pointercancel',()=>{state.gesture=null;drawOverlay()});
function undo(){const p=current();if(!p||!p.undo.length)return;p.redo.push(structuredClone(p.edits));p.edits=p.undo.pop();updateUI()}
function redo(){const p=current();if(!p||!p.redo.length)return;p.undo.push(structuredClone(p.edits));p.edits=p.redo.pop();updateUI()}
$('undoButton').addEventListener('click',undo);$('redoButton').addEventListener('click',redo);
$('clearEdits').addEventListener('click',()=>{const p=current();if(!p||!p.edits.length)return;if(!confirm('清空本页全部编辑？可以用撤销恢复。'))return;p.undo.push(structuredClone(p.edits));p.edits=[];p.redo=[];updateUI()});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?redo():undo()}if(e.key==='ArrowRight')goPage(state.index+1);if(e.key==='ArrowLeft')goPage(state.index-1)});
async function exportOne(){const p=current();if(!p)return;busy(true,'生成 PNG','按原始 TARGET 分辨率合成...');try{const [t,s]=await Promise.all([bitmap(currentTarget()),currentSrc()?bitmap(currentSrc()):null]);const c=compose(t,s,p);const blob=await canvasBlob(c);c.width=0;c.height=0;download(blob,safeName(currentTarget().name).replace(/\.[^.]+$/,'')+'_folirina.png');toast('本页 PNG 已生成')}catch(e){toast(`导出失败：${e.message}`);console.error(e)}finally{busy(false)}}
$('exportPage').addEventListener('click',exportOne);
$('exportAll').addEventListener('click',async()=>{
 if(!state.targets.length)return;busy(true,'导出整册','正在生成 PNG 文件...');try{
 const zip=new JSZip(),dir=zip.folder('Folirina_Output');let count=0;
 for(let i=0;i<state.pairs.length;i++){
  const p=state.pairs[i];if(p.skip)continue;
  $('busyDetail').textContent=`合成页面 ${i+1}/${state.pairs.length}`;
  const t=await bitmap(state.targets[p.tgt]),s=p.src===null?null:await bitmap(state.sources[p.src]);
  const c=compose(t,s,p),blob=await canvasBlob(c);c.width=0;c.height=0;
  dir.file(`${String(i+1).padStart(4,'0')}_${safeName(state.targets[p.tgt].name).replace(/\.[^.]+$/,'')}.png`,blob);count++;
  await new Promise(r=>setTimeout(r,0));
 }
 if(!count)throw Error('所有页面均被跳过');
 $('busyDetail').textContent='压缩 ZIP...';const out=await zip.generateAsync({type:'blob',compression:'STORE'},meta=>{$('busyDetail').textContent=`压缩 ZIP ${Math.round(meta.percent)}%`});download(out,'Folirina_Web_Export.zip');toast(`已导出 ${count} 页 PNG，按 TARGET 原分辨率保存`);
 }catch(e){toast(`导出失败：${e.message}`);console.error(e)}finally{busy(false)}
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
 const pairs=data.pairs.map((p,i)=>{if(p.tgt!==i||!(p.src===null||Number.isInteger(p.src)&&p.src>=0&&p.src<newSources.length)||!Array.isArray(p.edits)||p.edits.length>5000||p.edits.some(e=>!'transfer erase clone restore'.split(' ').includes(e.kind)||!'rect ellipse brush'.split(' ').includes(e.shape))||(p.affine!==null&&p.affine!==undefined&&(!Array.isArray(p.affine)||p.affine.length!==6||p.affine.some(n=>typeof n!=='number'||!Number.isFinite(n)||Math.abs(n)>100000))))throw Error('配对数据错误');return {...defaultPair(i,newSources.length),...p,undo:[],redo:[]}});
 releaseItems(state.sources);releaseItems(state.targets);state.sources=newSources;state.targets=newTargets;state.pairs=pairs;state.index=Math.min(Math.max(0,data.index||0),pairs.length-1);newSources=[];newTargets=[];updateUI();toast(`工作区加载成功，共 ${pairs.length} 页`);
 }catch(err){releaseItems(newSources);releaseItems(newTargets);toast(`加载失败：${err.message}`);console.error(err)}finally{busy(false)}
});
if('serviceWorker' in navigator && location.protocol.startsWith('http'))navigator.serviceWorker.register('./sw.js').catch(()=>{});
updateUI();
// For smoke/regression tests only: no privileged APIs or network actions.
window.__folirinaTest={state,composeCurrent:async()=>{const p=current();return compose(await bitmap(currentTarget()),currentSrc()?await bitmap(currentSrc()):null,p)},goPage};
