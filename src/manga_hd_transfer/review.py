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

_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Manga HD Transfer Review</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#151515;color:#eee}header{padding:12px 18px;background:#222;position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center}button,select,input,textarea{background:#2c2c2c;color:#eee;border:1px solid #555;border-radius:6px;padding:6px}main{padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}.panel{background:#202020;padding:8px;border-radius:8px;min-width:0}.panel img{width:100%;height:auto;display:block}.canvasWrap{position:relative;width:100%}.canvasWrap img{width:100%;display:block}.canvasWrap canvas{position:absolute;left:0;top:0;width:100%;height:100%;opacity:.42;filter:sepia(1) saturate(15) hue-rotate(315deg)}table{width:100%;border-collapse:collapse;margin-top:12px;background:#202020}th,td{border-bottom:1px solid #444;padding:7px;text-align:left}td input[type=text]{width:95%}.bad{color:#ff6b6b}.ok{color:#7ee787}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.small{font-size:12px;color:#aaa}</style>
</head><body><header><strong>出版级迁移 Review</strong><select id="pageSel"></select><button onclick="loadPage()">载入</button><span id="status"></span></header>
<main><div class="grid"><div class="panel"><h3>旧中文版</h3><img id="source"></div><div class="panel"><h3>高清日文 / 清字 Mask</h3><div class="canvasWrap"><img id="target"><canvas id="mask"></canvas></div><div class="toolbar"><button onclick="mode='paint'">Mask 增加</button><button onclick="mode='erase'">Mask 擦除</button><label>笔刷 <input id="brush" type="range" min="2" max="80" value="18"></label><button onclick="saveMask()">保存 Mask</button></div></div><div class="panel"><h3>当前输出</h3><img id="final"></div></div>
<div class="toolbar"><button onclick="saveReview()">保存文字/匹配</button><button onclick="applyReview()">应用复核并重新生成</button><input id="notes" placeholder="复核备注"><select id="reviewStatus"><option>reviewed</option><option>needs_work</option><option>approved</option></select></div>
<table><thead><tr><th>应用</th><th>旧中文译文</th><th>目标区域</th><th>原匹配</th></tr></thead><tbody id="rows"></tbody></table><pre id="qa" class="small"></pre></main>
<script>
let current='', project=null, mode='paint', drawing=false;
const canvas=document.getElementById('mask'), ctx=canvas.getContext('2d');
async function api(url,opt){let r=await fetch(url,opt);if(!r.ok)throw new Error(await r.text());return r.json()}
async function init(){let p=await api('/api/pages');let s=document.getElementById('pageSel');p.pages.forEach(x=>{let o=document.createElement('option');o.value=x;o.textContent=x;s.appendChild(o)});if(p.pages.length){current=p.pages[0];s.value=current;await loadPage()}}
async function loadPage(){current=document.getElementById('pageSel').value;project=await api('/api/project?page='+encodeURIComponent(current));document.getElementById('source').src='/asset?page='+encodeURIComponent(current)+'&name=source';document.getElementById('target').src='/asset?page='+encodeURIComponent(current)+'&name=target_original.png';document.getElementById('final').src='/asset?page='+encodeURIComponent(current)+'&name=best_final&t='+Date.now();document.getElementById('qa').textContent=JSON.stringify(project.qa,null,2);buildRows();await loadMask()}
function buildRows(){let tbody=document.getElementById('rows');tbody.innerHTML='';let targets=project.target_units||[];let match={};(project.matches||[]).filter(x=>x.relation==='one_to_one').forEach(x=>match[x.source_unit_id]=x.target_unit_id);let auto=new Set((project.meta?.auto_applied_match_ids||[]).map(x=>x.split('->')[0]));(project.source_units||[]).forEach(s=>{let tr=document.createElement('tr');let opts=targets.map(t=>`<option value="${t.id}" ${match[s.id]===t.id?'selected':''}>${t.id} · ${t.kind}</option>`).join('');tr.innerHTML=`<td><input class="accept" data-id="${s.id}" type="checkbox" ${auto.has(s.id)?'checked':''}></td><td><input class="txt" data-id="${s.id}" type="text" value="${esc(s.text||'')}"><div class="small">${s.id} conf=${(s.confidence||0).toFixed(2)}</div></td><td><select class="match" data-id="${s.id}">${opts}</select></td><td>${match[s.id]||'—'}</td>`;tbody.appendChild(tr)})}
function esc(s){return s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;')}
async function loadMask(){let img=new Image();img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.clearRect(0,0,canvas.width,canvas.height);let off=document.createElement('canvas');off.width=canvas.width;off.height=canvas.height;let o=off.getContext('2d');o.drawImage(img,0,0);let d=o.getImageData(0,0,off.width,off.height);for(let i=0;i<d.data.length;i+=4){let v=d.data[i];d.data[i]=255;d.data[i+1]=0;d.data[i+2]=0;d.data[i+3]=v>10?255:0}ctx.putImageData(d,0,0)};img.src='/asset?page='+encodeURIComponent(current)+'&name=best_mask&t='+Date.now()}
function pos(e){let r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}
function stroke(e){if(!drawing)return;let [x,y]=pos(e);let rad=+document.getElementById('brush').value*canvas.width/canvas.getBoundingClientRect().width;ctx.save();ctx.globalCompositeOperation=mode==='erase'?'destination-out':'source-over';ctx.fillStyle='rgba(255,0,0,1)';ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);ctx.fill();ctx.restore()}
canvas.onpointerdown=e=>{drawing=true;canvas.setPointerCapture(e.pointerId);stroke(e)};canvas.onpointermove=stroke;canvas.onpointerup=e=>drawing=false;
async function saveMask(){let body={page:current,png:canvas.toDataURL('image/png')};await api('/api/save-mask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});setStatus('Mask 已保存',true)}
async function saveReview(){let text={},matches={},accepted=[];document.querySelectorAll('.txt').forEach(x=>text[x.dataset.id]=x.value);document.querySelectorAll('.match').forEach(x=>matches[x.dataset.id]=x.value);document.querySelectorAll('.accept').forEach(x=>{if(x.checked)accepted.push(x.dataset.id)});let body={page:current,text_overrides:text,match_overrides:matches,accepted_source_units:accepted,status:document.getElementById('reviewStatus').value,notes:document.getElementById('notes').value};await api('/api/save-review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});setStatus('复核已保存',true)}
async function applyReview(){await saveReview();let r=await api('/api/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({page:current})});document.getElementById('final').src='/asset?page='+encodeURIComponent(current)+'&name=best_final&t='+Date.now();setStatus('已重新生成 '+r.final,true)}
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
                n=int(self.headers.get("Content-Length","0")); return json.loads(self.rfile.read(n) or b"{}")

            def do_GET(self):
                u=urlparse(self.path); q=parse_qs(u.query)
                try:
                    if u.path=="/":
                        data=_HTML.encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
                    if u.path=="/api/pages": return self._json({"pages":server.pages()})
                    if u.path=="/api/project":
                        p=server.page_dir(q.get("page",[""])[0]); return self._json(load_json(p/"project.json"))
                    if u.path=="/asset":
                        p=server.page_dir(q.get("page",[""])[0]); name=q.get("name",[""])[0]
                        project=load_json(p/"project.json")
                        if name=="source": path=Path(project["pair"]["source_path"])
                        elif name=="best_final": path=(p/"final_reviewed.png") if (p/"final_reviewed.png").exists() else (p/"final.png")
                        elif name=="best_mask": path=(p/"manual_clear_mask.png") if (p/"manual_clear_mask.png").exists() else (p/"clear_mask.png")
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
                        payload={k:body.get(k) for k in ("text_overrides","match_overrides","accepted_source_units","status","notes")}; save_json(p/"review_overrides.json",payload); return self._json({"ok":True})
                    if self.path=="/api/save-mask":
                        raw=str(body.get("png","")).split(",",1)[-1]; arr=np.frombuffer(base64.b64decode(raw),np.uint8); rgba=cv2.imdecode(arr,cv2.IMREAD_UNCHANGED)
                        if rgba is None: raise ValueError("invalid PNG")
                        if rgba.ndim==3 and rgba.shape[2]==4: mask=(rgba[:,:,3]>20).astype(np.uint8)*255
                        else: mask=(cv2.cvtColor(rgba,cv2.COLOR_BGR2GRAY)>20).astype(np.uint8)*255
                        target=cv2.imread(str(p/"target_original.png"));
                        if target is None or mask.shape!=target.shape[:2]: raise ValueError("mask size mismatch")
                        cv2.imwrite(str(p/"manual_clear_mask.png"),mask); return self._json({"ok":True})
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
