from __future__ import annotations

import base64
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .config import PipelineConfig
from .io_utils import load_json, save_json
from .review_apply import apply_review_page
from .result_state import resolve_result_state
from .schema_compat import as_dict, normalize_project, normalize_route_meta, merge_review_overrides


def _freshest_page_final(page_dir: Path) -> Path:
    state = resolve_result_state(page_dir)
    return state.current if state.current is not None else page_dir / "final.png"

_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Manga HD Transfer Review</title>
<style>
:root{--bg:#f3f7fc;--surface:#fff;--surface2:#f8fbff;--line:#e3ebf5;--text:#17243b;--muted:#8794a8;--blue:#6d9cf4;--blue2:#4f83e8;--blueSoft:#e8f1ff;--green:#66ae93;--red:#d97782}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;margin:0;background:var(--bg);color:var(--text);font-size:13px}header{height:58px;padding:0 18px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;display:flex;gap:9px;align-items:center;backdrop-filter:blur(16px)}header strong{font-size:14px;margin-right:8px}button,select,input,textarea{font:inherit;background:#fff;color:var(--text);border:1px solid #d4e0ee;border-radius:9px;padding:7px 10px;outline:none}button{cursor:pointer}button:hover{background:#f2f7ff;border-color:#bfd4f4}button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}button.primary:hover{background:var(--blue2)}main{padding:14px}.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.panel{background:var(--surface);padding:10px;border:1px solid var(--line);border-radius:12px;min-width:0;box-shadow:0 1px 2px rgba(61,89,128,.03)}.panel h3{font-size:12px;margin:2px 2px 9px;color:#53627a}.panel img{width:100%;height:auto;display:block;border-radius:8px;background:#f5f8fc}.canvasWrap{position:relative;width:100%}.canvasWrap img{width:100%;display:block}.canvasWrap canvas{position:absolute;left:0;top:0;width:100%;height:100%;opacity:.34;filter:sepia(1) saturate(8) hue-rotate(165deg)}table{width:100%;border-collapse:separate;border-spacing:0;margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left}th{background:#f2f7ff;color:#53627a;font-size:11px}tr:last-child td{border-bottom:0}td input[type=text]{width:96%}.bad{color:var(--red)}.ok{color:var(--green)}.toolbar{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0;align-items:center}.small{font-size:11px;color:var(--muted)}#qa{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;max-height:220px;overflow:auto}input[type=range]{accent-color:var(--blue)}
</style>
</head><body><header><strong>出版级迁移 Review</strong><select id="pageSel"></select><button onclick="loadPage()">载入</button><span id="status"></span></header>
<main><div class="grid"><div class="panel"><h3>旧中文版 · 译文证据</h3><img id="source"></div><div class="panel"><h3 id="maskTitle">高清日文 · 清字 Mask</h3><div class="canvasWrap"><img id="target"><canvas id="mask"></canvas></div><div class="toolbar"><button onclick="mode='paint'">增加 Mask</button><button onclick="mode='erase'">擦除 Mask</button><label class="small">笔刷 <input id="brush" type="range" min="2" max="80" value="18"></label><button onclick="saveMask()">保存 Mask</button></div></div><div class="panel"><h3>出版预览 · 当前输出</h3><img id="final"></div></div>
<div class="toolbar"><button onclick="saveReview()">保存文字 / 匹配</button><button class="primary" onclick="applyReview()">应用复核并重新生成</button><button onclick="forcePage('force_direct_patch')">强制 Direct</button><button onclick="forcePage('force_mask_replace')">强制 Mask</button><input id="notes" placeholder="复核备注"><select id="reviewStatus"><option>reviewed</option><option>needs_work</option><option>approved</option></select></div>
<table><thead><tr><th>应用</th><th>旧中文译文</th><th>目标区域</th><th>原匹配</th><th>强制动作</th></tr></thead><tbody id="rows"></tbody></table><pre id="qa" class="small"></pre></main>
<script>
let current='', project=null, mode='paint', drawing=false, pageForceAction='';
const canvas=document.getElementById('mask'), ctx=canvas.getContext('2d');
async function api(url,opt){let r=await fetch(url,opt);if(!r.ok)throw new Error(await r.text());return r.json()}
async function init(){let p=await api('/api/pages');let s=document.getElementById('pageSel');p.pages.forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;s.appendChild(o)});if(p.pages.length){current=p.pages[0];s.value=current;await loadPage()}}
async function loadPage(){current=document.getElementById('pageSel').value;project=await api('/api/project?page='+encodeURIComponent(current));document.getElementById('source').src='/asset?page='+encodeURIComponent(current)+'&name=source';document.getElementById('target').src='/asset?page='+encodeURIComponent(current)+'&name=target_original.png';document.getElementById('final').src='/asset?page='+encodeURIComponent(current)+'&name=best_final&t='+Date.now();document.getElementById('qa').textContent=JSON.stringify(project.qa,null,2);document.getElementById('maskTitle').textContent=(project.meta?.transfer_mode==='direct_patch'?'高清日文 · Direct 贴图区域':(project.meta?.transfer_mode==='mask_replace'?'高清日文 · 蒙版迁移区域':(project.meta?.transfer_mode==='auto'?'高清日文 · 自动路线写入区域':'高清日文 · 清字 Mask')));buildRows();await loadMask()}
function buildRows(){let tbody=document.getElementById('rows');tbody.innerHTML='';let targets=project.target_units||[];let match={};(project.matches||[]).filter(x=>x.relation==='one_to_one').forEach(x=>match[x.source_unit_id]=x.target_unit_id);let auto=new Set((project.meta?.auto_applied_match_ids||[]).map(x=>x.split('->')[0]));(project.source_units||[]).forEach(s=>{let tr=document.createElement('tr');let opts=targets.map(t=>`<option value="${t.id}" ${match[s.id]===t.id?'selected':''}>${t.id} · ${t.kind}</option>`).join('');let top=(project.meta?.matching_diagnostics?.top_candidates?.[s.id]||[]).slice(0,3).map(x=>`${x.target_unit_id}:${Number(x.cost).toFixed(3)}`).join(' · ');tr.innerHTML=`<td><input class="accept" data-id="${s.id}" type="checkbox" ${auto.has(s.id)?'checked':''}></td><td><input class="txt" data-id="${s.id}" type="text" value="${esc(s.text||'')}"><div class="small">${s.id} conf=${(s.confidence||0).toFixed(2)}${top?'<br>Top: '+esc(top):''}</div></td><td><select class="match" data-id="${s.id}">${opts}</select></td><td>${match[s.id]||'—'}</td><td><select class="unitAction" data-id="${s.id}"><option value="auto">自动</option><option value="force_match">强制匹配</option><option value="skip_unit">跳过</option></select></td>`;tbody.appendChild(tr)})}
function esc(s){return s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;')}
async function loadMask(){let img=new Image();img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.clearRect(0,0,canvas.width,canvas.height);let off=document.createElement('canvas');off.width=canvas.width;off.height=canvas.height;let o=off.getContext('2d');o.drawImage(img,0,0);let d=o.getImageData(0,0,off.width,off.height);for(let i=0;i<d.data.length;i+=4){let v=d.data[i];d.data[i]=255;d.data[i+1]=0;d.data[i+2]=0;d.data[i+3]=v>10?255:0}ctx.putImageData(d,0,0)};img.src='/asset?page='+encodeURIComponent(current)+'&name=best_mask&t='+Date.now()}
function pos(e){let r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}
function stroke(e){if(!drawing)return;let [x,y]=pos(e);let rad=+document.getElementById('brush').value*canvas.width/canvas.getBoundingClientRect().width;ctx.save();ctx.globalCompositeOperation=mode==='erase'?'destination-out':'source-over';ctx.fillStyle='rgba(255,0,0,1)';ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);ctx.fill();ctx.restore()}
canvas.onpointerdown=e=>{drawing=true;canvas.setPointerCapture(e.pointerId);stroke(e)};canvas.onpointermove=stroke;canvas.onpointerup=e=>drawing=false;
async function saveMask(){let body={page:current,png:canvas.toDataURL('image/png')};await api('/api/save-mask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});setStatus('Mask 已保存',true)}
async function saveReview(){let text={},matches={},accepted=[],unitActions={};document.querySelectorAll('.txt').forEach(x=>text[x.dataset.id]=x.value);document.querySelectorAll('.match').forEach(x=>matches[x.dataset.id]=x.value);document.querySelectorAll('.accept').forEach(x=>{if(x.checked)accepted.push(x.dataset.id)});document.querySelectorAll('.unitAction').forEach(x=>{if(x.value&&x.value!=='auto')unitActions[x.dataset.id]=x.value});let body={page:current,text_overrides:text,match_overrides:matches,accepted_source_units:accepted,unit_actions:unitActions,page_force_action:pageForceAction,status:document.getElementById('reviewStatus').value,notes:document.getElementById('notes').value};await api('/api/save-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});setStatus('复核已保存'+(pageForceAction?' · '+pageForceAction:''),true)}
async function forcePage(action){pageForceAction=action;await applyReview();pageForceAction=''}
async function applyReview(){await saveReview();let r=await api('/api/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({page:current})});project=await api('/api/project?page='+encodeURIComponent(current));document.getElementById('final').src='/asset?page='+encodeURIComponent(current)+'&name=best_final&t='+Date.now();document.getElementById('qa').textContent=JSON.stringify(project.qa,null,2);setStatus('已重新生成 '+r.final,true)}
function setStatus(s,ok){let e=document.getElementById('status');e.textContent=s;e.className=ok?'ok':'bad'}
init().catch(e=>setStatus(e.message,false));
</script></body></html>'''


class ReviewServer:
    def __init__(self, output_dir: str | Path, config: PipelineConfig | None = None) -> None:
        self.root = Path(output_dir).resolve()
        self.pages_root = self.root / "pages"
        self.config = config or PipelineConfig()
        if not self.pages_root.exists():
            raise FileNotFoundError(f"No pages directory: {self.pages_root}")

    def pages(self) -> list[str]:
        return sorted(p.name for p in self.pages_root.iterdir() if p.is_dir() and (p / "project.json").exists())

    def page_dir(self, name: str) -> Path:
        p = (self.pages_root / name).resolve()
        if p.parent != self.pages_root or not p.is_dir():
            raise FileNotFoundError(name)
        return p

    def handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def _json(self, payload, status=200):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)

            def _body(self):
                n=int(self.headers.get("Content-Length","0"))
                payload=json.loads(self.rfile.read(n) or b"{}")
                return as_dict(payload)

            def do_GET(self):
                u=urlparse(self.path); q=parse_qs(u.query)
                try:
                    if u.path=="/":
                        data=_HTML.encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
                    if u.path=="/api/pages": return self._json({"pages":server.pages()})
                    if u.path=="/api/project":
                        p=server.page_dir(q.get("page",[""])[0]); return self._json(normalize_project(load_json(p/"project.json")))
                    if u.path=="/asset":
                        p=server.page_dir(q.get("page",[""])[0]); name=q.get("name",[""])[0]
                        project=normalize_project(load_json(p/"project.json"))
                        if name=="source": path=Path(project["pair"]["source_path"])
                        elif name=="best_final": path=_freshest_page_final(p)
                        elif name=="best_mask":
                            meta=as_dict(project.get("meta")); mode=str(meta.get("transfer_mode","reletter"))
                            direct_used=bool(normalize_route_meta(meta.get("direct_patch")).get("used"))
                            if direct_used and (p/"direct_patch_regions.png").exists():
                                path=(p/"manual_direct_patch_regions.png") if (p/"manual_direct_patch_regions.png").exists() else (p/"direct_patch_regions.png")
                            elif mode in {"auto","mask_replace","hybrid"} and (p/"mask_transfer_mask.png").exists():
                                path=(p/"manual_transfer_mask.png") if (p/"manual_transfer_mask.png").exists() else (p/"mask_transfer_mask.png")
                            else:
                                path=(p/"manual_clear_mask.png") if (p/"manual_clear_mask.png").exists() else (p/"clear_mask.png")
                        else:
                            if "/" in name or "\\" in name or name.startswith("."): raise FileNotFoundError(name)
                            path=p/name
                        if not path.exists(): raise FileNotFoundError(path)
                        data=path.read_bytes(); ctype=mimetypes.guess_type(path.name)[0] or "application/octet-stream"; self.send_response(200); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
                    self.send_error(404)
                except Exception as e: self._json({"error":str(e)},400)

            def do_POST(self):
                try:
                    body=self._body(); p=server.page_dir(str(body.get("page","")))
                    if self.path=="/api/save-review":
                        review_path=p/"review_overrides.json"
                        payload=merge_review_overrides(load_json(review_path) if review_path.exists() else {}, body)
                        save_json(review_path,payload); return self._json({"ok":True})
                    if self.path=="/api/save-mask":
                        raw=str(body.get("png","")).split(",",1)[-1]; arr=np.frombuffer(base64.b64decode(raw),np.uint8); rgba=cv2.imdecode(arr,cv2.IMREAD_UNCHANGED)
                        if rgba is None: raise ValueError("invalid PNG")
                        if rgba.ndim==3 and rgba.shape[2]==4: mask=(rgba[:,:,3]>20).astype(np.uint8)*255
                        else: mask=(cv2.cvtColor(rgba,cv2.COLOR_BGR2GRAY)>20).astype(np.uint8)*255
                        target=cv2.imread(str(p/"target_original.png"));
                        if target is None or mask.shape!=target.shape[:2]: raise ValueError("mask size mismatch")
                        project=normalize_project(load_json(p/"project.json")); meta=as_dict(project.get("meta")); mode=str(meta.get("transfer_mode","reletter"))
                        direct_used=bool(normalize_route_meta(meta.get("direct_patch")).get("used"))
                        if direct_used and (p/"direct_patch_regions.png").exists(): name="manual_direct_patch_regions.png"
                        elif mode in {"auto","mask_replace","hybrid"} and (p/"mask_transfer_mask.png").exists(): name="manual_transfer_mask.png"
                        else: name="manual_clear_mask.png"
                        cv2.imwrite(str(p/name),mask); return self._json({"ok":True,"mask":name})
                    if self.path=="/api/apply":
                        final=apply_review_page(p,server.config); return self._json({"ok":True,"final":str(final)})
                    self.send_error(404)
                except Exception as e: self._json({"error":str(e)},400)
        return Handler


def serve_review(output_dir: str | Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, config: PipelineConfig | None = None) -> None:
    app = ReviewServer(output_dir, config)
    httpd = ThreadingHTTPServer((host, port), app.handler())
    url = f"http://{host}:{port}/"
    print(f"Review server: {url}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
