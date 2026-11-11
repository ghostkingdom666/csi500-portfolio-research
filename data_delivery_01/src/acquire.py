"""Online acquisition. Run sample before all; successful snapshots are never replaced."""
from common import *
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd

def universe():
    t=pd.read_excel(ROOT/'raw/references/csi_current.xls',dtype=str)
    a=pd.DataFrame({'source_code':t.iloc[:,4].str.zfill(6), 'name':t.iloc[:,5],
                    'exchange_name':t.iloc[:,7], 'list_reference_date':t.iloc[:,0]})
    a['exchange']=a.exchange_name.map({'上海证券交易所':'SH','深圳证券交易所':'SZ'})
    assert a.exchange.notna().all(), 'unexpected exchange'
    a['asset_id']=a.source_code+'.'+a.exchange
    a['list_reference_date']=pd.to_datetime(a.list_reference_date,format='%Y%m%d').dt.strftime('%Y-%m-%d')
    a=a.sort_values('asset_id').reset_index(drop=True)
    assert len(a)==500 and a.asset_id.nunique()==500
    return a

def config():
    path=ROOT/'snapshot_config.json'
    if path.exists():return read_json(path)
    at=datetime.now(HK)
    # Calendar builder subsequently checks this candidate against official holiday rules.
    latest=(at.date()-timedelta(days=1) if at.hour<15 else at.date()).isoformat()
    a=universe()
    samples=[a.loc[a.source_code.str.startswith(p),'asset_id'].iloc[0] for p in ['600','000','300']]
    c={'data_version':'csi500_'+at.strftime('%Y%m%dT%H%M%S')+'_current_v1',
       'created_at':at.isoformat(timespec='seconds'), 'requested_start':'2025-01-01',
       'requested_end':at.date().isoformat(), 'closed_date_candidate':latest,
       'anchor_date':'2024-12-31', 'estimation_end':'2026-06-30',
       'test_start':'2026-07-01', 'subset_size':100,'sample_assets':samples,
       'pool_type':'current_constituents_retrospective',
       'historical_pool_verified':False,'historical_fallback_authorized':True,
       'price_source':'eastmoney','adjustment':'hfq',
       'universe_source':'https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls'}
    write_json(path,c);return c

def one_asset(asset,c):
    errors=[]
    for kind,fqt in [('raw',0),('hfq',2)]:
        try: fetch(f'prices/{asset}/{kind}.json',em_url(asset,fqt,end=c['requested_end'].replace('-','')),validate_em)
        except Exception as exc:errors.append({'asset_id':asset,'kind':kind,'error':str(exc)})
    code,ex=asset.split('.')
    u='https://push2.eastmoney.com/api/qt/stock/get?'+urllib.parse.urlencode({
        'secid':('1' if ex=='SH' else '0')+'.'+code,'fields':'f57,f58,f84,f85,f189'})
    try:
        def valid(b):assert json.loads(b).get('data'), 'empty metadata'
        fetch(f'metadata/{asset}.json',u,valid)
    except Exception as exc:errors.append({'asset_id':asset,'kind':'metadata','error':str(exc)})
    time.sleep(.12)
    return errors

def crosscheck(asset,c):
    code,ex=asset.split('.');sym=ex.lower()+code
    for adj in ['', 'hfq']:
        p=f'{sym},day,2024-12-31,{c["requested_end"]},1000,{adj}'
        u='https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?'+urllib.parse.urlencode({'param':p})
        def valid(b):
            j=json.loads(b);assert j.get('code')==0 and j.get('data',{}).get(sym), 'empty Tencent response'
        fetch(f'crosscheck/{asset}_{adj or "raw"}.json',u,valid)
    # Cash/stock distributions are separate evidence, not inferred from price changes alone.
    p={'reportName':'RPT_SHAREBONUS_DET','columns':'ALL','filter':f'(SECURITY_CODE="{code}")',
       'sortColumns':'EX_DIVIDEND_DATE','sortTypes':'-1','pageSize':500,'pageNumber':1,'source':'WEB','client':'WEB'}
    u='https://datacenter-web.eastmoney.com/api/data/v1/get?'+urllib.parse.urlencode(p)
    try:
        def valid(b):assert json.loads(b).get('result',{}).get('data') is not None, 'no corporate action response'
        fetch(f'events/{asset}.json',u,valid)
    except Exception as exc:print('EVENT UNAVAILABLE',asset,str(exc),flush=True)

def sample_check(c):
    rows=[]
    for asset in c['sample_assets']:
        one_asset(asset,c);crosscheck(asset,c)
        raw=em_frame((ROOT/f'raw/prices/{asset}/raw.json').read_bytes())
        hfq=em_frame((ROOT/f'raw/prices/{asset}/hfq.json').read_bytes())
        code,ex=asset.split('.');sym=ex.lower()+code
        j=read_json(ROOT/f'raw/crosscheck/{asset}_raw.json')['data'][sym]
        tx=pd.DataFrame(j.get('day',[]));assert len(tx)>0
        tx=tx.iloc[:,:6];tx.columns=['date','open_tx','close_tx','high_tx','low_tx','volume_lots_tx']
        for col in tx.columns[1:]:tx[col]=pd.to_numeric(tx[col],errors='raise')
        m=raw.merge(tx,on='date',validate='one_to_one')
        close_diff=float((m.close-m.close_tx).abs().max());vol_diff=float((m.volume_lots-m.volume_lots_tx).abs().max())
        assert close_diff<=.011 and vol_diff<=1, f'raw source mismatch: {asset}'
        assert raw.date.max()==hfq.date.max(), 'adjustment coverage mismatch'
        assert hfq.close.gt(0).all(), 'non-positive adjusted prices'
        assert raw['amount'].ge(0).all() and raw.volume_lots.ge(0).all()
        j=read_json(ROOT/f'raw/crosscheck/{asset}_hfq.json')['data'][sym]
        tr=pd.DataFrame(j.get('hfqday',j.get('day',[]))).iloc[:,:6]
        tr.columns=['date','open_tx','close_tx','high_tx','low_tx','volume_lots_tx']
        tr.close_tx=pd.to_numeric(tr.close_tx)
        h=hfq.merge(tr[['date','close_tx']],on='date',validate='one_to_one').sort_values('date')
        dif=(h.close.pct_change(fill_method=None)-h.close_tx.pct_change(fill_method=None)).abs()
        # EM publishes two decimals, Tencent three: compare rounding intervals,
        # rather than rejecting genuine matches using an arbitrary return tolerance.
        adjusted_price_diff=float((h.close-h.close_tx).abs().max())
        prev=h.close.shift(1);delta=.00550001
        rounding_bound=delta/prev+(h.close+delta)*delta/(prev*(prev-delta))
        same_definition=(adjusted_price_diff<=.00550001 and
                         (dif.iloc[1:]<=rounding_bound.iloc[1:]+1e-12).all())
        maxret=float(dif.max())
        # Different vendor adjustment conventions are a disclosed limitation. The
        # primary hfq series is used consistently; never splice the Tencent series.
        h['primary_return']=h.close.pct_change(fill_method=None)
        h['check_return']=h.close_tx.pct_change(fill_method=None)
        h['return_diff']=dif
        (ROOT/'evidence').mkdir(exist_ok=True)
        csv_out(h[['date','close','close_tx','primary_return','check_return','return_diff']],
                ROOT/f'evidence/{asset}_adjustment_comparison.csv')
        rows.append({'asset_id':asset,'rows':len(raw),'first_date':raw.date.min(),'last_date':raw.date.max(),
                     'crosscheck_rows':len(m),'max_raw_close_diff_yuan':close_diff,
                     'max_volume_diff_lots':vol_diff,'max_adjusted_price_diff_yuan':adjusted_price_diff,
                     'max_adjusted_return_diff':maxret,'rounding_interval_check_passed':bool(same_definition),
                     'adjustment_crosscheck_status':'matches_with_rounding' if same_definition else 'vendor_convention_disagreement_disclosed'})
        print('SAMPLE PASSED',rows[-1],flush=True)
    write_json(ROOT/'sample_validation.json',{'passed':True,
        'passed_scope':'raw_prices_volume_required_fields_positive_primary_hfq',
        'adjusted_cross_provider_equivalence_claimed':False,'results':rows})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['sample','all'],default='sample')
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();c=config()
    if args.stage=='sample':sample_check(c);return
    assert read_json(ROOT/'sample_validation.json')['passed'], 'run sample first'
    assets=universe().asset_id.tolist();errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(one_asset,a,c):a for a in assets}
        for i,f in enumerate(as_completed(fs),1):
            errors.extend(f.result())
            if i%25==0 or i==len(assets):print('ACQUIRE',i,'/',len(assets),'failures',len(errors),flush=True)
    try:fetch('index/000905.json',em_url('000905.SH',0,end=c['requested_end'].replace('-','')),validate_em)
    except Exception as e:errors.append({'asset_id':'index000905','kind':'index','error':str(e)})
    write_json(ROOT/'acquisition_failures.json',errors)
    print('ACQUISITION FINISHED',len(errors),'failures',flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
