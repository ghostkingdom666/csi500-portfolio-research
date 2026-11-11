"""Acquire date-specific suspension evidence for gaps, excluding pre-listing days."""
from common import *
from acquire import config,universe
from trading_calendar import calendar
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd

def gap_requests():
    c=config();cal=calendar(c);days=set(cal.loc[cal.is_open.eq(1),'date']);targets={};gap_assets=[]
    for asset in universe().asset_id:
        p=ROOT/f'raw/prices/{asset}/raw.json'
        if not p.exists():continue
        f=em_frame(p.read_bytes());mp=ROOT/f'raw/metadata/{asset}.json'
        listing=''
        if mp.exists():
            v=str(read_json(mp)['data'].get('f189',''))
            if len(v)==8 and v.isdigit() and v!='00000000':listing=f'{v[:4]}-{v[4:6]}-{v[6:]}'
        missing=days-set(f.date)
        missing.update(f.loc[f.volume_lots.eq(0),'date'])
        missing={d for d in missing if not listing or d>=listing}
        if missing:gap_assets.append(asset)
        for d in missing:targets.setdefault(d,[]).append(asset)
    return targets,gap_assets

def main():
    targets,assets=gap_requests();write_json(ROOT/'gap_evidence_requests.json',dict(sorted(targets.items())))
    def get_day(day):
        p={'reportName':'RPT_CUSTOM_SUSPEND_DATA_INTERFACE','columns':'ALL','pageSize':500,'pageNumber':1,
           'sortColumns':'SUSPEND_START_DATE','sortTypes':'-1','source':'WEB','client':'WEB',
           'filter':f'(MARKET="全部")(DATETIME=\'{day}\')'}
        u='https://datacenter-web.eastmoney.com/api/data/v1/get?'+urllib.parse.urlencode(p)
        try:
            def valid(b):assert json.loads(b).get('success'), 'invalid suspension response'
            j=json.loads(fetch(f'status/{day}.json',u,valid));total=len(j['result']['data'])
            for page in range(2,int(j['result']['pages'])+1):
                p['pageNumber']=page
                pu='https://datacenter-web.eastmoney.com/api/data/v1/get?'+urllib.parse.urlencode(p)
                extra=json.loads(fetch(f'status/{day}_page{page}.json',pu,valid))
                total+=len(extra['result']['data'])
            assert total==j['result']['count'], 'incomplete suspension pagination'
            return {'date':day,'ok':True,'records':total,'pages':j['result']['pages']}
        except Exception as e:return {'date':day,'ok':False,'error':str(e)}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        fs=[ex.submit(get_day,d) for d in sorted(targets)]
        for i,f in enumerate(as_completed(fs),1):
            results.append(f.result())
            if i%20==0:print('STATUS',i,'/',len(targets),flush=True)
    # Corporate actions for gapped assets support valuation checks across suspensions.
    def get_events(asset):
        p={'reportName':'RPT_SHAREBONUS_DET','columns':'ALL','filter':f'(SECURITY_CODE="{asset[:6]}")',
           'sortColumns':'EX_DIVIDEND_DATE','sortTypes':'-1','pageSize':500,'pageNumber':1,'source':'WEB','client':'WEB'}
        u='https://datacenter-web.eastmoney.com/api/data/v1/get?'+urllib.parse.urlencode(p)
        try:
            def valid(b):assert json.loads(b).get('success'), 'invalid corporate action response'
            j=json.loads(fetch(f'events/{asset}.json',u,valid))
            assert j['result']['pages']<=1, 'event pagination needed'
            return {'asset_id':asset,'ok':True}
        except Exception as e:return {'asset_id':asset,'ok':False,'error':str(e)}
    with ThreadPoolExecutor(max_workers=4) as ex:events=list(ex.map(get_events,assets))
    write_json(ROOT/'enrichment_results.json',{'status_requests':sorted(results,key=lambda r:r['date']),
        'corporate_action_requests':events})
    print('ENRICHMENT FINISHED',len(targets),'dates',len(assets),'assets; failures',
          sum(not r['ok'] for r in results+events),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
