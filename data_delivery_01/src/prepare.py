"""Acquire reference evidence and inspect the candidate universe before prices."""
from common import *
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import pandas as pd

SOURCES={
 'csi_current.xls':'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls',
 'sse_2025.html':'https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20241223_10767110.shtml',
 'sse_2026.html':'https://www.sse.com.cn/disclosure/dealinstruc/closed/',
 'akshare_price_docs.html':'https://akshare.akfamily.xyz/data/stock/stock_quote.html',
 'akshare_index_docs.html':'https://akshare.akfamily.xyz/data/index/index.html',
 'baostock_api.md':'https://raw.githubusercontent.com/zxygithub/baostock/master/docs/pythonAPI.md',
}

def main():
    def get(kv):
        k,u=kv
        try:
            b=fetch('references/'+k,u); return {'file':k,'ok':True,'bytes':len(b)}
        except Exception as e: return {'file':k,'ok':False,'error':str(e)}
    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(get,SOURCES.items()): print(r,flush=True)
    df=pd.read_excel(BytesIO((ROOT/'raw/references/csi_current.xls').read_bytes()), dtype=str)
    print('CSI TABLE',df.shape,df.columns.tolist(),flush=True)
    print(df.head(3).to_string(index=False),flush=True)
    # Baostock weekly snapshot is corroborating historical evidence, not an official announcement.
    import socket
    socket.setdefaulttimeout(12)
    import baostock as bs
    history={'queried_at':now(),'query_date':'2026-06-30','provider':'baostock','is_official_announcement':False}
    try:
        lg=bs.login();history['login_code']=lg.error_code;history['login_msg']=lg.error_msg
        if lg.error_code!='0': raise RuntimeError(lg.error_msg)
        rs=bs.query_zz500_stocks(date='2026-06-30')
        history['error_code']=rs.error_code;history['error_msg']=rs.error_msg
        history['fields']=rs.fields;history['rows']=[]
        while rs.error_code=='0' and rs.next(): history['rows'].append(rs.get_row_data())
        bs.logout()
    except Exception as exc: history['exception']=repr(exc)
    write_json(ROOT/'raw/references/baostock_history_probe.json',history)
    print('HISTORY PROBE', {k:v for k,v in history.items() if k!='rows'},'ROWS',len(history.get('rows',[])),flush=True)

if __name__=='__main__': main()
