"""Bounded session cache for deterministic manual region replay."""
from collections import OrderedDict
import copy
import hashlib
import json
import threading

_entries=OrderedDict()
_lock=threading.Lock()
_bytes=0
_limit=128*1024*1024


def replay_transparent(action,current,target,source,project,row,cfg,*,page_dir,repaired_background_mask):
    global _bytes
    digest=hashlib.sha256()
    for array in (current,target,source,repaired_background_mask):
        if array is None:digest.update(b'none');continue
        digest.update(str((array.shape,str(array.dtype))).encode())
        digest.update(array.tobytes())
    config=cfg.model_dump(mode='json') if hasattr(cfg,'model_dump') else repr(cfg)
    identity=copy.deepcopy(project)
    # These publication receipts change on every save and do not participate
    # in registration, text extraction or compositing.
    for field in ('delivery_export','review_sync'):
        identity.get('meta',{}).pop(field,None)
    digest.update(json.dumps([str(page_dir),identity,row,config],sort_keys=True,default=str).encode())
    key=digest.digest()
    with _lock:
        hit=_entries.get(key)
        if hit is not None:
            _entries.move_to_end(key)
            return tuple(a.copy() for a in hit[:3])+(copy.deepcopy(hit[3]),)
    result=action(current,target,source,project,row,cfg,page_dir=page_dir,
                  repaired_background_mask=repaired_background_mask)
    if not result[3].get('success'):return result
    size=sum(a.nbytes for a in result[:3])
    if size>_limit:return result
    saved=tuple(a.copy() for a in result[:3])+(copy.deepcopy(result[3]),)
    with _lock:
        old=_entries.pop(key,None)
        if old is not None:_bytes-=sum(a.nbytes for a in old[:3])
        while _entries and _bytes+size>_limit:
            _,old=_entries.popitem(last=False);_bytes-=sum(a.nbytes for a in old[:3])
        _entries[key]=saved;_bytes+=size
    return result
