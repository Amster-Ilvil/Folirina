#!/usr/bin/env python3
"""Local-only Folirina Web native engine. Serves docs + ORIGINAL Python image algorithms.

Usage from the Folirina repository root (with desktop dependencies installed):
    python tools/web_bridge.py
Open http://127.0.0.1:8765/ — NOT the public GitHub Pages origin.
The public site remains 100% browser-local. No uploads leave this computer.
"""
from __future__ import annotations

import argparse
import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
DOCS = ROOT / "docs"

MAX_BODY = 48 * 1024 * 1024
MAX_DIMENSION = 12000
MAX_PIXELS = 30_000_000
MAX_EDITS = 200
ALLOWED_ACTIONS = {"transparent": "region_transparent", "hole": "region_hole_reveal",
                   "texture": "region_direct_patch", "precision": "region_precise_mask"}


def _images(data: str):
    import cv2
    import numpy as np
    if not isinstance(data, str) or not data.startswith("data:image/") or "," not in data:
        raise ValueError("只接受带 MIME 类型的图片 Data URL")
    mime, encoded = data.split(",", 1)
    if ";base64" not in mime or not any(mime.startswith("data:image/" + x) for x in ("png", "jpeg", "webp", "bmp")):
        raise ValueError("图片编码类型不支持")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as e:
        raise ValueError("图片 Base64 无效") from e
    if len(raw) > 24 * 1024 * 1024:
        raise ValueError("单张压缩图片超过 24 MB")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片解码失败")
    h, w = image.shape[:2]
    if w > MAX_DIMENSION or h > MAX_DIMENSION or w * h > MAX_PIXELS:
        raise ValueError("图片分辨率超过本地服务保护上限")
    return image


def _matrix(data):
    import numpy as np
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape == (6,):
        a,b,c,d,e,f = [float(v) for v in arr]
        arr = np.asarray([[a,c,e],[b,d,f],[0,0,1]], np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > 1e6:
        raise ValueError("变换矩阵不合法")
    if abs(np.linalg.det(arr)) < 1e-10:
        raise ValueError("配准矩阵不可逆")
    return arr


def _edit_selection(edit: dict, w: int, h: int):
    import numpy as np
    import cv2
    shape = edit.get('shape')
    mask = np.zeros((h, w), np.uint8)
    if shape == 'brush':
        points = edit.get('points', [])
        if not isinstance(points,list) or len(points)>10000:
            raise ValueError('画笔路径无效')
        pts=[]
        for point in points:
            x,y=float(point['x']),float(point['y'])
            if not np.isfinite(x+y):raise ValueError('画笔坐标无效')
            pts.append([max(0,min(w-1,round(x))),max(0,min(h-1,round(y)))])
        if not pts:raise ValueError('空画笔路径')
        brush = max(2,min(240,round(float(edit.get('brush') or 24))))
        vec=np.array(pts,np.int32)
        if len(vec)>1: cv2.polylines(mask,[vec],False,255,brush,lineType=cv2.LINE_8)
        for x,y in (vec if len(vec)==1 else [vec[0],vec[-1]]):
            cv2.circle(mask,(int(x),int(y)),max(1,brush//2),255,-1)
    elif shape in ('rect','ellipse'):
        a,b=edit.get('start') or {},edit.get('end') or {}
        x0=max(0,min(w-1,round(min(float(a['x']),float(b['x'])))))
        x1=max(0,min(w,round(max(float(a['x']),float(b['x'])))))
        y0=max(0,min(h-1,round(min(float(a['y']),float(b['y'])))))
        y1=max(0,min(h,round(max(float(a['y']),float(b['y'])))))
        if x1<=x0 or y1<=y0:raise ValueError('空选区')
        if shape=='rect':mask[y0:y1,x0:x1]=255
        else:cv2.ellipse(mask,((x0+x1-1)//2,(y0+y1-1)//2),((x1-x0)//2,(y1-y0)//2),0,0,360,255,-1)
    else:raise ValueError('选区形状不支持')
    if cv2.countNonZero(mask)<3:raise ValueError('空选区')
    return mask


def _selection_spec(mask):
    import cv2
    import numpy as np
    from manga_hd_transfer.region_selection import bbox_from_mask
    bbox=bbox_from_mask(mask)
    cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnt:raise ValueError('无法读取画笔轮廓')
    largest=max(cnt,key=cv2.contourArea)
    # Use desktop's own freehand selection path to preserve selection authority.
    # Allow many vertices for curved brushes, but bound maliciously huge paths.
    poly=largest.reshape(-1,2)  # CHAIN_APPROX_SIMPLE preserves raster contour exactly
    if len(poly)>5000:raise ValueError('画笔轮廓太复杂，请分成较短的笔画')
    if len(poly)<3: raise ValueError('画笔区域过窄')
    return {'schema':'folirina.region_selection.v1','kind':'freehand',
            'bbox':bbox,'points':poly.astype(int).tolist()}


def process_register(source, target):
    from manga_hd_transfer.registration import register_images
    from manga_hd_transfer.config import RegistrationConfig
    result=register_images(source,target,RegistrationConfig(backend='opencv',allow_model_downloads=False))
    H=result.matrix
    return {'matrix':H.tolist(),'affine':[float(H[0,0]),float(H[1,0]),float(H[0,1]),float(H[1,1]),float(H[0,2]),float(H[1,2])],
            'confidence':float(result.confidence),'inliers':int(result.num_matches),
            'medianError':float(result.reprojection_error),'method':str(result.method),
            'lowConfidence':bool(result.confidence<.55)}


def process_render(source,target,matrix,edits):
    import numpy as np
    import cv2
    from manga_hd_transfer.config import PipelineConfig
    from manga_hd_transfer.region_composite import apply_region_action
    if not isinstance(edits,list) or len(edits)>MAX_EDITS:raise ValueError('编辑记录数量超过安全上限')
    H=_matrix(matrix)
    project={'registration':{'matrix':H.tolist()},'meta':{'transfer_mode':'direct_patch'}}
    cfg=PipelineConfig()
    out=target.copy();receipts=[]
    for i,edit in enumerate(edits):
        if not isinstance(edit,dict):raise ValueError('编辑记录无效')
        mask=_edit_selection(edit,target.shape[1],target.shape[0])
        kind=edit.get('kind')
        if kind=='transfer':
            mode=ALLOWED_ACTIONS.get(edit.get('mode') or 'hole')
            if not mode:raise ValueError('不支持的迁移模式')
            spec=_selection_spec(mask)
            row={'mode':mode,'id':f'web-{i+1}',
                 'selection_spec':spec,'target_bbox':spec['bbox']}
            before=out.copy()
            candidate,_,_,audit=apply_region_action(out,target,source,project,row,cfg)
            if not audit.get('success'):raise ValueError(f'第{i+1}笔无有效结果: {audit.get("reason", "未知") }')
            if candidate.shape!=out.shape:raise ValueError('原生区域结果尺寸不匹配')
            # Even newer desktop renderers may claim a whole verified bubble.
            # The manual WEB brush, however, is explicitly limited to its own
            # painted authority. Never modify a pixel outside that authority.
            changed_before=np.any(candidate!=before,axis=2)
            outside=int(np.count_nonzero(changed_before & (mask==0)))
            out[mask>0]=candidate[mask>0]
            actual=int(np.count_nonzero(np.any(out!=before,axis=2)))
            receipts.append({'index':i+1,'mode':mode,'changed':actual,
                             'clippedOutsideSelection':outside,
                             'clearPixels':audit.get('target_clear_pixels',0)})
        elif kind=='restore':
            out[mask>0]=target[mask>0];receipts.append({'index':i+1,'mode':'restore'})
        elif kind=='erase':
            paper=max(0,min(255,int(edit.get('paper',255))));out[mask>0]=paper
            receipts.append({'index':i+1,'mode':'erase'})
        elif kind=='clone':
            offset=edit.get('cloneOffset') or {}
            dx=int(round(float(offset.get('x',0))));dy=int(round(float(offset.get('y',0))))
            ys,xs=np.where(mask>0)
            out[ys,xs]=target[np.clip(ys+dy,0,target.shape[0]-1),np.clip(xs+dx,0,target.shape[1]-1)]
            receipts.append({'index':i+1,'mode':'clone'})
        else:raise ValueError('不支持的编辑种类')
    ok,png=cv2.imencode('.png',out)
    if not ok:raise ValueError('PNG 编码失败')
    return {'png':'data:image/png;base64,'+base64.b64encode(png.tobytes()).decode(),
            'receipts':receipts,'width':int(target.shape[1]),'height':int(target.shape[0])}


def process_warp(source,target,matrix):
    import cv2
    import numpy as np
    H=_matrix(matrix)
    out=cv2.warpPerspective(source,H,(target.shape[1],target.shape[0]),flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    ok,png=cv2.imencode('.png',out)
    if not ok:raise ValueError('预览层生成失败')
    return {'png':'data:image/png;base64,'+base64.b64encode(png.tobytes()).decode()}


def serve(host='127.0.0.1',port=8765,quiet=False):
    if host!='127.0.0.1':raise ValueError('安全限制：仅允许监听 127.0.0.1')
    # Fail before advertising engine health if desktop dependencies are missing.
    try:
        import numpy
        import cv2
        from manga_hd_transfer.registration import register_images
        from manga_hd_transfer.region_composite import apply_region_action
    except ImportError as exc:
        raise RuntimeError('缺少 Folirina 桌面 Python 环境依赖，请先安装项目依赖：'+str(exc)) from exc
    token=secrets.token_urlsafe(32)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(DOCS),**kwargs)
        def _host_ok(self):
            origin_host=self.headers.get('Host','')
            if not origin_host:return False
            try:
                hostname=urlsplit('http://'+origin_host).hostname
                port_value=urlsplit('http://'+origin_host).port
            except ValueError:return False
            if hostname not in ('127.0.0.1','localhost'):return False
            if port_value is not None and port_value!=self.server.server_port:return False
            # Reject cross-origin browser requests even if a malicious site
            # attempts to send user content with a local-only host header.
            origin=self.headers.get('Origin')
            if origin:
                try:
                    parsed=urlsplit(origin)
                    if parsed.scheme!='http' or parsed.hostname!=hostname or parsed.port!=self.server.server_port:
                        return False
                except ValueError:return False
            return True
        def log_message(self,fmt,*args):
            if not quiet:super().log_message(fmt,*args)
        def end_headers(self):
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.send_header('Cross-Origin-Resource-Policy','same-origin')
            super().end_headers()
        def _reply(self,obj,status=200):
            raw=json.dumps(obj,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self._host_ok():self._reply({'error':'invalid Host'},403);return
            if urlsplit(self.path).path=='/api/engine/health':
                self._reply({'ok':True,'engine':'Folirina Python 原生区域引擎','token':token,
                             'modes':list(ALLOWED_ACTIONS),'registration':'OpenCV SIFT/ORB + RANSAC',
                             'modelRequired':False});return
            return super().do_GET()
        def do_POST(self):
            if not self._host_ok():self._reply({'error':'invalid Host'},403);return
            path=urlsplit(self.path).path
            if path not in ('/api/engine/register','/api/engine/render','/api/engine/warp'):
                self._reply({'error':'unknown endpoint'},404);return
            if self.headers.get('X-Folirina-Token')!=token:
                self._reply({'error':'未授权操作，请从本地工作台重新打开'},403);return
            length=int(self.headers.get('Content-Length','0'))
            if length<2 or length>MAX_BODY:
                self._reply({'error':'请求太大或为空'},413);return
            try:
                req=json.loads(self.rfile.read(length))
                source,target=_images(req['source']),_images(req['target'])
                if path.endswith('register'):data=process_register(source,target)
                elif path.endswith('warp'):data=process_warp(source,target,req.get('matrix'))
                else:data=process_render(source,target,req.get('matrix'),req.get('edits',[]))
                self._reply(data)
            except Exception as e:
                self._reply({'error':f'{type(e).__name__}: {str(e)[:300]}'},422)
    httpd=ThreadingHTTPServer((host,port),Handler)
    print(f'Folirina 原生网页工作台：http://{host}:{port}/  (仅本机使用)')
    try:httpd.serve_forever()
    except KeyboardInterrupt:pass
    finally:httpd.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Serve Folirina Web with original Python processing on 127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();serve(port=args.port)
