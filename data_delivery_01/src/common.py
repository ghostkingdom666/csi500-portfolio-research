"""Snapshot I/O shared by online acquisition and deterministic offline build."""
from __future__ import annotations
import hashlib, json, sys, time, threading, urllib.request, urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
HK = timezone(timedelta(hours=8))
LOCK = threading.Lock()

def now():
    return datetime.now(HK).isoformat(timespec='seconds')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def log(data):
    with LOCK:
        with (ROOT/'requests.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, sort_keys=True)+'\n')

def fetch(rel, url, validator=None, timeout=25, attempts=3):
    """Immutable successful responses; resume reads the existing snapshot."""
    path = ROOT/'raw'/rel
    if path.exists():
        content = path.read_bytes()
        if validator: validator(content)
        return content
    path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(1, attempts+1):
        at = now()
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0', 'Referer':'https://quote.eastmoney.com/'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                meta = {'url':url, 'retrieved_at':at, 'attempt':attempt,
                        'http_status':response.status, 'headers':dict(response.headers),
                        'path':str(path.relative_to(ROOT)).replace('\\','/'),
                        'sha256':hashlib.sha256(content).hexdigest(), 'bytes':len(content)}
            # Persist failed server responses too, outside the successful raw path.
            if validator:
                try: validator(content)
                except Exception:
                    failed = ROOT/'raw'/'failed_responses'/(rel.replace('/','_')+f'.attempt{attempt}')
                    failed.parent.mkdir(parents=True, exist_ok=True); failed.write_bytes(content)
                    raise
            path.write_bytes(content)
            write_json(str(path)+'.meta.json', meta)
            log(dict(meta, status='ok'))
            return content
        except Exception as exc:
            error = repr(exc)
            log({'url':url,'retrieved_at':at,'attempt':attempt,'path':rel,'status':'failed','error':error})
            if attempt < attempts: time.sleep(attempt)
    raise RuntimeError(f'{rel}: {error}')

def csv_out(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n',
              float_format='%.17g', na_rep='')

def em_url(asset_id, adjust, start='20241231', end='20261111'):
    code, market = asset_id.split('.')
    p = dict(secid=('1' if market=='SH' else '0')+'.'+code, klt=101,
             fqt=adjust, beg=start, end=end, lmt=1000,
             fields1='f1,f2,f3,f4,f5,f6', fields2='f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61')
    return 'https://push2his.eastmoney.com/api/qt/stock/kline/get?'+urllib.parse.urlencode(p)

def validate_em(b):
    j=json.loads(b)
    assert j.get('rc')==0 and j.get('data') and j['data'].get('klines'), 'empty or invalid price response'

def em_frame(b):
    import pandas as pd
    rows=json.loads(b)['data']['klines']
    cols=['date','open','close','high','low','volume_lots','amount','amplitude_pct','change_pct','change','turnover_pct']
    df=pd.DataFrame([r.split(',') for r in rows], columns=cols)
    for c in cols[1:]: df[c]=pd.to_numeric(df[c], errors='coerce')
    return df
