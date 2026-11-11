"""Resolve an actual historical status omission using issuer/venue announcements."""
from common import *
from trading_calendar import calendar
import re
from io import BytesIO
from pypdf import PdfReader

ANNOUNCEMENTS={
 '002155_suspend_20260112.pdf':'https://static.cninfo.com.cn/finalpage/2026-01-12/1224929055.PDF',
 '002155_resume_20260126.pdf':'https://disc.static.szse.cn/disc/disk03/finalpage/2026-01-26/ed783f50-c513-4f81-9713-20bde02af77c.PDF',
}

def main():
    evidence=[];texts=[]
    for name,url in ANNOUNCEMENTS.items():
        b=fetch('status_announcements/'+name,url,lambda b: b.startswith(b'%PDF') or (_ for _ in ()).throw(ValueError('not a PDF')))
        text='\n'.join(p.extract_text() or '' for p in PdfReader(BytesIO(b)).pages)
        (ROOT/'raw/status_announcements'/(name+'.txt')).write_text(text,encoding='utf-8')
        texts.append(re.sub(r'\s+','',text))
        evidence.append({'path':'raw/status_announcements/'+name,'url':url,'sha256':sha(ROOT/'raw/status_announcements'/name)})
    assert '002155' in texts[0] and '2026年1月12日' in texts[0] and '停牌' in texts[0]
    assert '002155' in texts[1] and '2026年1月26日' in texts[1] and '复牌' in texts[1]
    cal=calendar(read_json(ROOT/'snapshot_config.json'))
    ds=cal.loc[cal.is_open.eq(1)&cal.date.ge('2026-01-12')&cal.date.lt('2026-01-26'),'date'].tolist()
    result={'asset_id':'002155.SZ','status':'confirmed_suspended','start_date':'2026-01-12',
        'resume_date':'2026-01-26','confirmed_suspension_dates':ds,'evidence':evidence,
        'reason':'issuer suspension and SZSE-hosted resumption notices; primary status interface omitted this earlier episode',
        'inference':'open dates between effective suspension start and exclusive resumption date; no price inference'}
    write_json(ROOT/'raw/status_announcements/verified_interval.json',result)
    print('ANNOUNCEMENTS VERIFIED',len(ds),'trading dates',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
