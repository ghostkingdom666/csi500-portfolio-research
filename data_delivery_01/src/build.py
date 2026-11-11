"""Deterministically rebuild the delivery from frozen local responses, without network."""
from common import *
from acquire import universe
from trading_calendar import calendar
import argparse, importlib.metadata, json
import numpy as np
import pandas as pd

def load_status():
    status={}
    for path in sorted((ROOT/'raw/status').glob('*.json')):
        if path.name.endswith('.meta.json'):continue
        day=path.stem.split('_')[0];j=read_json(path)
        for r in (j.get('result') or {}).get('data',[]):
            asset=r.get('SECUCODE','');start=(r.get('SUSPEND_START_DATE') or '')[:10]
            end=(r.get('SUSPEND_END_TIME') or '')[:10]
            if start and start<=day and (not end or day<=end):
                status[(asset,day)]={'status':'confirmed_suspended',
                    'source':str(path.relative_to(ROOT)).replace('\\','/'),
                    'reason':r.get('SUSPEND_REASON',''),'start':start,'end':end}
    verified=ROOT/'raw/status_announcements/verified_interval.json'
    if verified.exists():
        j=read_json(verified)
        for e in j['evidence']:assert sha(ROOT/e['path'])==e['sha256'], 'announcement hash mismatch'
        for day in j['confirmed_suspension_dates']:
            status[(j['asset_id'],day)]={'status':'confirmed_suspended',
                'source':'raw/status_announcements/verified_interval.json',
                'reason':j['reason'],'start':j['start_date'],'end':j['resume_date']}
    return status

def load_events(asset):
    p=ROOT/f'raw/events/{asset}.json'
    if not p.exists():return None
    j=read_json(p)
    return (j.get('result') or {}).get('data',[])

def dedup_frame(path,kind,asset,diagnostics):
    if not path.exists():return pd.DataFrame(columns=['date','open','close','high','low','volume_lots','amount'])
    f=em_frame(path.read_bytes());original=len(f)
    exact=int(f.duplicated().sum());f=f.drop_duplicates()
    conflicts=set(f.loc[f.duplicated('date',keep=False),'date'])
    diagnostics.append({'asset_id':asset,'kind':kind,'original_rows':original,'exact_duplicates':exact,
                        'conflicting_dates':len(conflicts)})
    f=f.drop_duplicates('date',keep='first').sort_values('date').set_index('date')
    f['key_conflict']=f.index.isin(conflicts)
    return f.reset_index()

def returns_from_prices(f):
    # Reindex BEFORE this calculation. Never use pandas implicit fill or skip gaps.
    p=f.close_adj.where(f.valuation_valid.eq(1))
    return p.div(p.shift(1)).sub(1)

def choose_subset(prices,returns,asset_ids,c):
    est=prices.loc[prices.date.between(c['requested_start'],c['estimation_end'])]
    rr=returns.loc[returns.date.between(c['requested_start'],c['estimation_end'])]
    T=len(rr);rows=[]
    for asset in asset_ids:
        q=est.loc[est.asset_id.eq(asset)]
        finite_returns=int(np.isfinite(rr[asset]).sum())
        amounts=np.isfinite(q.amount)&q.amount.ge(0)
        n_amount=int(amounts.sum())
        bad=int(q.key_conflict.sum())+int(q.quality_invalid.sum())
        eligible=bool(finite_returns==T and n_amount==T and bad==0)
        reason='eligible' if eligible else ';'.join(x for x,v in [
            ('incomplete_estimation_returns',finite_returns!=T),('incomplete_or_invalid_estimation_amount',n_amount!=T),
            ('unresolved_estimation_price_conflict_or_invalid_data',bad>0)] if v)
        rows.append({'asset_id':asset,'estimation_T':T,'finite_return_days':finite_returns,
                     'valid_amount_days':n_amount,'estimation_issue_count':bad,'eligible':int(eligible),
                     'mean_daily_amount':float(q.amount.mean()) if n_amount else np.nan,'reason':reason})
    audit=pd.DataFrame(rows).set_index('asset_id')
    ranked=audit.loc[audit.eligible.eq(1)].reset_index().sort_values(['mean_daily_amount','asset_id'],ascending=[False,True])
    audit['liquidity_rank']=pd.Series(range(1,len(ranked)+1),index=ranked.asset_id,dtype='Int64')
    selected=ranked.asset_id.iloc[:c['subset_size']].tolist()
    audit['selected']=audit.index.isin(selected).astype(int)
    audit.loc[audit.eligible.eq(1)&audit.selected.eq(0),'reason']='eligible_below_liquidity_cutoff'
    audit.loc[audit.selected.eq(1),'reason']='selected_by_estimation_liquidity'
    return audit.reset_index(),selected

def assemble(c,cal):
    a=universe();ids=a.asset_id.tolist();days=cal.loc[cal.is_open.eq(1),'date'].tolist()
    status=load_status();pieces=[];diagnostics=[];off_calendar=[];asset_meta=[]
    for n,asset in enumerate(ids,1):
        r=dedup_frame(ROOT/f'raw/prices/{asset}/raw.json','raw',asset,diagnostics)
        h=dedup_frame(ROOT/f'raw/prices/{asset}/hfq.json','hfq',asset,diagnostics)
        for kind,t in [('raw',r),('hfq',h)]:
            if len(t):
                off=t.loc[~t.date.isin(days)].copy()
                if len(off):off['asset_id']=asset;off['kind']=kind;off_calendar.append(off)
        f=r.set_index('date').reindex(days)
        f.index.name='date';f=f.reset_index()
        for col in ['open','close','high','low','volume_lots','amount']:
            if col not in f:f[col]=np.nan
        f=f.rename(columns={'open':'open_raw','close':'close_raw','high':'high_raw','low':'low_raw'})
        hp=h.set_index('date').reindex(days)
        f['close_adj']=hp['close'].to_numpy() if 'close' in hp else np.nan
        f['key_conflict']=(f.get('key_conflict',pd.Series(False,index=f.index)).fillna(False).astype(bool).to_numpy() |
                           hp.get('key_conflict',pd.Series(False,index=days)).fillna(False).astype(bool).to_numpy()).astype(int)
        f['volume']=f.volume_lots*100
        mp=ROOT/f'raw/metadata/{asset}.json';listing='';meta_source='unknown'
        if mp.exists():
            d=read_json(mp).get('data') or {};v=str(d.get('f189',''))
            if len(v)==8 and v.isdigit() and int(v[:4])>=1900:
                listing=f'{v[:4]}-{v[4:6]}-{v[6:]}';meta_source=str(mp.relative_to(ROOT)).replace('\\','/')
        f['asset_id']=asset;f['tradable_status']='missing_unknown';f['status_source']='no_verified_status_evidence'
        f['valuation_method']='missing';f['valuation_valid']=0
        observed=f.close_raw.notna()&f.close_adj.notna()&f.volume.gt(0)
        invalid=(f.close_raw.le(0)|f.close_adj.le(0)|f.volume.lt(0)|f.amount.lt(0)|
                 f.high_raw.lt(f.low_raw)|f.high_raw.lt(f.close_raw)|f.low_raw.gt(f.close_raw)|
                 f.high_raw.lt(f.open_raw)|f.low_raw.gt(f.open_raw))
        f['quality_invalid']=invalid.astype(int)
        good=observed&~invalid&f.key_conflict.eq(0)
        f.loc[good,'tradable_status']='observed_trading'
        f.loc[good,'status_source']='eastmoney_daily_bar_positive_volume'
        f.loc[good,'valuation_method']='vendor_close'
        f.loc[good,'valuation_valid']=1
        if listing:
            pre=f.date.lt(listing)
            f.loc[pre,'tradable_status']='before_listing';f.loc[pre,'status_source']=meta_source
        # Confirmed suspensions are evidenced date by date. Only test-period valuation
        # may carry forward a verified mark, with no unverified corporate action.
        events=load_events(asset)
        event_days=set((e.get('EX_DIVIDEND_DATE') or '')[:10] for e in (events or []))
        for ix in f.index:
            key=(asset,f.at[ix,'date'])
            if key not in status:continue
            if good.iloc[ix]:continue  # intraday suspension with a positive-volume bar
            s=status[key];f.at[ix,'tradable_status']='confirmed_suspended';f.at[ix,'status_source']=s['source']
            if f.at[ix,'date']>=c['test_start'] and ix>0 and events is not None and f.at[ix,'date'] not in event_days:
                if f.at[ix-1,'valuation_valid']==1:
                    for col in ['close_raw','close_adj']:f.at[ix,col]=f.at[ix-1,col]
                    f.at[ix,'volume']=0.;f.at[ix,'amount']=0.
                    f.at[ix,'valuation_method']='verified_suspension_carry';f.at[ix,'valuation_valid']=1
        f['return']=returns_from_prices(f)
        f['missing_reason']=''
        f.loc[f.valuation_valid.eq(0),'missing_reason']=f.loc[f.valuation_valid.eq(0),'tradable_status']
        f.loc[f.valuation_valid.eq(1)&f['return'].isna(),'missing_reason']='previous_trading_day_price_missing'
        f.loc[f.key_conflict.eq(1),'missing_reason']='conflicting_duplicate_key'
        f.loc[f.quality_invalid.eq(1),'missing_reason']='invalid_price_volume_or_ohlc'
        f.loc[f.date.eq(c['anchor_date']),'missing_reason']='anchor_no_return_required'
        f['source']='eastmoney';f['adjustment']='hfq';f['is_anchor']=f.date.lt(c['requested_start']).astype(int)
        raw_meta=ROOT/f'raw/prices/{asset}/raw.json.meta.json';adj_meta=ROOT/f'raw/prices/{asset}/hfq.json.meta.json'
        f['retrieved_at']=read_json(raw_meta)['retrieved_at'] if raw_meta.exists() else ''
        f['adj_retrieved_at']=read_json(adj_meta)['retrieved_at'] if adj_meta.exists() else ''
        cols=['date','asset_id','open_raw','high_raw','low_raw','close_raw','close_adj','volume','amount',
              'tradable_status','status_source','valuation_method','valuation_valid','key_conflict','quality_invalid',
              'return','missing_reason','source','adjustment','retrieved_at','adj_retrieved_at','is_anchor']
        pieces.append(f[cols])
        asset_meta.append({'asset_id':asset,'listing_date':listing,'listing_source':meta_source,
                           'delisting_date':'','delisting_status':'unknown','delisting_source':'not_verified',
                           'raw_first_date':r.date.min() if len(r) else '',
                           'raw_last_date':r.date.max() if len(r) else '',
                           'observed_raw_rows':len(r),'observed_hfq_rows':len(h),
                           'download_status':'ok' if len(r) and len(h) else 'failed'})
        if n%100==0:print('BUILD',n,'/',len(ids),flush=True)
    prices=pd.concat(pieces,ignore_index=True).sort_values(['date','asset_id']).reset_index(drop=True)
    assets=a.merge(pd.DataFrame(asset_meta),on='asset_id',validate='one_to_one')
    assets['index_code']='000905';assets['pool_type']=c['pool_type']
    assets['announcement_date']='';assets['availability_date']='';assets['effective_date']=''
    assets['list_retrieved_at']=read_json(ROOT/'raw/references/csi_current.xls.meta.json')['retrieved_at']
    assets['list_source']=c['universe_source']
    wide=prices.pivot(index='date',columns='asset_id',values='return').reindex(columns=ids)
    wide=wide.loc[wide.index>=c['requested_start']].reset_index()
    return prices,assets,wide,pd.DataFrame(diagnostics),off_calendar

def corporate_action_evidence(prices,c):
    records=[]
    for asset in c['sample_assets']:
        f=prices.loc[prices.asset_id.eq(asset)].set_index('date').sort_index()
        f['prev_raw']=f.close_raw.shift(1);f['prev_adj']=f.close_adj.shift(1)
        for e in load_events(asset) or []:
            day=(e.get('EX_DIVIDEND_DATE') or '')[:10]
            if not (c['requested_start']<=day<=c['closed_date_candidate']) or day not in f.index:continue
            q=f.loc[day];div=float(e.get('PRETAX_BONUS_RMB') or 0)/10
            bonus=float(e.get('BONUS_IT_RATIO') or 0)/10
            records.append({'asset_id':asset,'date':day,'plan':e.get('IMPL_PLAN_PROFILE',''),
                'notice_date':(e.get('NOTICE_DATE') or '')[:10],'cash_dividend_per_share':div,
                'bonus_shares_per_share':bonus,'prev_raw_close':q.prev_raw,'raw_close':q.close_raw,
                'raw_return':q.close_raw/q.prev_raw-1,'prev_adj_close':q.prev_adj,'adj_close':q.close_adj,
                'adjusted_return':q['return'],
                'cash_and_bonus_arithmetic_reference':(q.close_raw*(1+bonus)+div)/q.prev_raw-1,
                'source':f'raw/events/{asset}.json',
                'conclusion':'event_aligned; supplier_hfq_proxy; not_strict_total_return'})
    return pd.DataFrame(records)

def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=read_json(ROOT/'snapshot_config.json');cal=calendar(c)
    prices,assets,returns,diagnostics,off=assemble(c,cal)
    ids=assets.asset_id.tolist();audit,selected=choose_subset(prices,returns,ids,c)
    assets=assets.merge(audit[['asset_id','eligible','selected','liquidity_rank','reason']],on='asset_id',validate='one_to_one')
    open_cal=cal.loc[cal.is_open.eq(1)].copy();last=open_cal.date.max()
    research=prices.loc[prices.is_anchor.eq(0)].copy()
    counts=prices.groupby('date').agg(valid_valuations=('valuation_valid','sum'),raw_bars=('volume',lambda s:int(s.gt(0).sum())))
    before=prices.tradable_status.eq('before_listing').groupby(prices.date).sum()
    complete=(counts.valid_valuations+before).eq(len(ids))
    latest_complete=counts.index[complete].max() if complete.any() else None
    raw_complete=counts.index[counts.raw_bars.eq(len(ids))].max() if counts.raw_bars.eq(len(ids)).any() else None
    # An auxiliary price index confirms that the holiday calendar matches observed market sessions.
    ip=ROOT/'raw/index/000905.json';index_calendar_match=False
    if ip.exists():
        idx=em_frame(ip.read_bytes());index_calendar_match=set(idx.date)==set(open_cal.date)
        idx['index_code']='000905';idx['return_type']='price_index'
        idx['daily_simple_return']=idx.close.pct_change(fill_method=None)
        csv_out(idx.sort_values('date'),out/'index_daily.csv')
    T=int(returns.date.le(c['estimation_end']).sum());N=len(selected)
    stats={
        'collection_count':len(ids),'download_success_count':int(assets.download_status.eq('ok').sum()),
        'eligible_count':int(audit.eligible.sum()),'model_subset_count':N,'estimation_T':T,
        'T_greater_than_N_plus_2':bool(T>N+2),'research_trading_days':len(returns),
        'test_trading_days_including_entry_date':int(returns.date.ge(c['test_start']).sum()),
        'raw_price_rows_before_deduplication':int(diagnostics.loc[diagnostics.kind.eq('raw'),'original_rows'].sum()),
        'raw_price_rows_after_deduplication':int(assets.observed_raw_rows.sum()),
        'hfq_rows_after_deduplication':int(assets.observed_hfq_rows.sum()),
        'aligned_price_rows_including_anchor':len(prices),'returns_rows':len(returns),'returns_columns_assets':len(ids),
        'missing_return_cells':int(returns[ids].isna().sum().sum()),
        'estimation_missing_return_cells':int(returns.loc[returns.date.le(c['estimation_end']),ids].isna().sum().sum()),
        'test_missing_return_cells':int(returns.loc[returns.date.ge(c['test_start']),ids].isna().sum().sum()),
        'selected_test_missing_return_cells':int(returns.loc[returns.date.ge(c['test_start']),selected].isna().sum().sum()),
        'exact_duplicates_removed':int(diagnostics.exact_duplicates.sum()),
        'conflicting_date_keys':int(diagnostics.conflicting_dates.sum()),
        'invalid_quality_rows':int(prices.quality_invalid.sum()),'off_calendar_records':int(sum(len(f) for f in off)),
        'unknown_status_rows':int(research.tradable_status.eq('missing_unknown').sum()),
        'confirmed_suspended_rows':int(research.tradable_status.eq('confirmed_suspended').sum()),
        'suspension_carry_rows':int(research.valuation_method.eq('verified_suspension_carry').sum()),
        'before_listing_rows':int(research.tradable_status.eq('before_listing').sum()),
        'latest_exchange_closed_date':last,'last_complete_trading_date':latest_complete,
        'last_date_with_raw_bars_for_all_500':raw_complete,
        'latest_day_raw_bars':int(counts.loc[last,'raw_bars']),
        'latest_day_verified_valuations':int(counts.loc[last,'valid_valuations']),
        'official_calendar_matches_index':bool(index_calendar_match),
    }
    blockers=[]
    if stats['unknown_status_rows']:blockers.append('unresolved_unknown_status_or_price_gaps')
    if stats['conflicting_date_keys'] or stats['invalid_quality_rows']:blockers.append('conflicting_or_invalid_primary_prices')
    if stats['selected_test_missing_return_cells']:blockers.append('selected_subset_test_returns_have_missing_values')
    if not index_calendar_match:blockers.append('official_calendar_not_corroborated_by_index')
    if N<c['subset_size']:blockers.append('insufficient_eligible_assets_for_requested_subset')
    if latest_complete!=last:blockers.append('latest_market_date_incomplete_valuation_coverage')
    limitations=['current_constituents_retrospective_not_historical_investable_universe',
        'survivorship_and_constituent_lookahead_bias', 'supplier_hfq_return_proxy_not_strict_total_return',
        '000009_SZ_adjusted_series_disagrees_between_vendors',
        'delisting_history_not_independently_verified', 'cross_member_signoff_pending']
    stats['technical_blockers']=blockers;stats['limitations']=limitations
    csv_out(prices,out/'prices_long.csv');csv_out(returns,out/'returns_wide.csv')
    csv_out(prices.loc[prices.asset_id.isin(c['sample_assets'])],out/'three_stock_sample.csv')
    mask=returns.copy();mask[ids]=mask[ids].isna().astype('uint8');csv_out(mask,out/'missing_mask.csv')
    csv_out(assets,out/'assets.csv');csv_out(open_cal,out/'calendar.csv');csv_out(cal,out/'calendar_all_days.csv')
    csv_out(audit,out/'selection_audit.csv')
    model_assets=assets.loc[assets.selected.eq(1)].copy();csv_out(model_assets,out/'model_assets.csv')
    model_ids=model_assets.asset_id.tolist();csv_out(returns[['date']+model_ids],out/'model_returns_wide.csv')
    csv_out(diagnostics,out/'cleaning_audit.csv')
    missing=research.loc[research['return'].isna(),['date','asset_id','missing_reason','tradable_status','status_source','valuation_method']]
    csv_out(missing,out/'missing_details.csv')
    csv_out(research.loc[research.tradable_status.ne('observed_trading'),
        ['date','asset_id','tradable_status','status_source','valuation_method','close_raw','close_adj']],out/'status_details.csv')
    if off:csv_out(pd.concat(off,ignore_index=True),out/'off_calendar_records.csv')
    else:csv_out(pd.DataFrame(columns=['date','asset_id','kind']),out/'off_calendar_records.csv')
    extreme=research.loc[research['return'].abs().gt(.25)].copy()
    csv_out(extreme,out/'extreme_return_review.csv')  # review flags never trigger clipping/deletion
    csv_out(corporate_action_evidence(prices,c),out/'corporate_action_checks.csv')
    decision=open_cal.loc[open_cal.date.le(c['estimation_end']),'date'].max()
    test_days=open_cal.loc[open_cal.date.ge(c['test_start']),'date'].tolist()
    periods={'estimation_start':returns.date.min(),'estimation_end':decision,
        'test_start':test_days[0] if test_days else None,'test_end':last,'decision_date':decision,
        'proposed_entry_date':test_days[0] if test_days else None,
        'proposed_first_return_date':test_days[1] if len(test_days)>1 else None,
        'entry_convention':'first July session close; first P&L is next close-to-close return',
        'entry_executed':False,'validation_split':None,'anchor_date':c['anchor_date'],
        'data_version':c['data_version']}
    write_json(out/'periods.json',periods);write_json(out/'quality_summary.json',stats)
    raw_records=[]
    for p in sorted((ROOT/'raw').rglob('*')):
        if p.is_file():raw_records.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    csv_out(pd.DataFrame(raw_records),out/'raw_inventory.csv')
    report(out,stats,c,assets,audit,extreme,periods)
    files={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['manifest.json','acceptance.json']}
    tracked_sources=list((ROOT/'src').glob('*.py'))+[ROOT/'pipeline.py',ROOT/'requirements.txt',ROOT/'snapshot_config.json']
    tracked_sources += [p for p in [ROOT/'README.md',ROOT/'数据需求与字段字典.md'] if p.exists()]
    source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(tracked_sources)}
    retrieval_times=[read_json(p)['retrieved_at'] for p in (ROOT/'raw').rglob('*.meta.json')]
    manifest={'data_version':c['data_version'],'retrieved_at':max(retrieval_times),
        'retrieval_started_at':min(retrieval_times),'snapshot_created_at':c['created_at'],
        'requested_start':c['requested_start'],'requested_end':c['requested_end'],
        'last_complete_trading_date':latest_complete,'latest_exchange_closed_date':last,
        'last_date_with_raw_bars_for_all_500':raw_complete,
        'last_complete_definition':'every candidate has observed or verified-suspension valuation; raw-bar date separately disclosed',
        'requested_end_not_closed_reason':'2026-11-11 market has not opened at snapshot acquisition',
        'pool_type':c['pool_type'],'historical_pool_verified':False,
        'source':'eastmoney','adjustment':'hfq','return_type':'supplier_adjusted_price_return_proxy',
        'units':{'raw_price':'CNY/share','adjusted_price':'supplier_hfq_CNY/share','volume':'shares','amount':'CNY','return':'decimal'},
        'subset_rule':'estimation-only complete finite returns/amount, no conflicts; mean amount desc, asset_id asc; top 100',
        'status':'usable_with_disclosed_limitations' if not blockers else 'delivery_with_technical_blockers',
        'technical_blockers':blockers,'limitations':limitations,'counts':stats,
        'files':files,'source_files':source_files,
        'environment':{'python':sys.version.split()[0],**{p:importlib.metadata.version(p) for p in ['numpy','pandas','openpyxl','xlrd']}},
        'sharing':'Publicly accessible does not establish redistribution permission; no explicit vendor redistribution license verified. Coursework review only; verify terms before public redistribution.'}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    return stats

def report(out,s,c,assets,audit,extreme,periods):
    samples=read_json(ROOT/'sample_validation.json')
    sample_lines='\n'.join(f'|{r["asset_id"]}|{r["rows"]}|{r["last_date"]}|{r["max_raw_close_diff_yuan"]:.6g}|{r["max_volume_diff_lots"]:.6g}|{r["max_adjusted_return_diff"]:.8g}|{r["adjustment_crosscheck_status"]}|' for r in samples['results'])
    text=f'''# 01任务卡数据质量与交接报告

版本：{c['data_version']}。这是**当前成分股回溯样本**，不是2026-06-30历史可投资股票池；已按用户批准降级。仅完成数据任务。

## 数量、日期与状态

|项目|实际结果|
|---|---|
|官方名单参考日|{assets.list_reference_date.iloc[0]}|
|收集名单 / 获取成功|{s['collection_count']} / {s['download_success_count']}|
|估计期质量合格 / 推荐子集|{s['eligible_count']} / {s['model_subset_count']}|
|估计期 T / 子集 N / T>N+2|{s['estimation_T']} / {s['model_subset_count']} / {s['T_greater_than_N_plus_2']}|
|研究区间交易日（不含锚点）|{s['research_trading_days']}|
|请求截止 / 最近已收盘日|{c['requested_end']} / {s['latest_exchange_closed_date']}|
|完整估值截至 / 全500只均有原始交易条截至|{s['last_complete_trading_date']} / {s['last_date_with_raw_bars_for_all_500']}|
|最新日原始交易条 / 已核实估值|{s['latest_day_raw_bars']} / {s['latest_day_verified_valuations']}|
|原始日线条数 / 对齐长表条数（含锚点）|{s['raw_price_rows_after_deduplication']} / {s['aligned_price_rows_including_anchor']}|
|收益矩阵行 × 资产列|{s['returns_rows']} × {s['returns_columns_assets']}|
|估计 / 检验 / 子集检验缺失收益单元|{s['estimation_missing_return_cells']} / {s['test_missing_return_cells']} / {s['selected_test_missing_return_cells']}|
|上市前 / 已核实停牌 / 未知状态行|{s['before_listing_rows']} / {s['confirmed_suspended_rows']} / {s['unknown_status_rows']}|
|检验期已核实停牌延续估值行|{s['suspension_carry_rows']}|
|完全重复删除 / 冲突键 / 无效质量行|{s['exact_duplicates_removed']} / {s['conflicting_date_keys']} / {s['invalid_quality_rows']}|
|官方日历与中证500指数日线日期集合一致|{s['official_calendar_matches_index']}|

抓取发生于香港时间 {c['created_at']} 起，2026-11-11尚未开市；逐次时间见原始文件旁的meta和requests.jsonl。最新原始交易条不足500不等同抓取落后：停牌估值及其证据单列，绝不声称每只股票都有当天成交。

## 三股与复权核查

|资产|日线条数含锚点|实际末日|两源原始收盘最大差（元）|成交量最大差（手）|复权日收益最大差|复权对照结果|
|---|---|---|---|---|---|---|
{sample_lines}

主源东方财富；腾讯仅核验。东方财富后复权价格通常保留两位小数、腾讯三位，白云机场与特锐德的差异落在价格舍入界限内。中国宝安的差异超过舍入范围，标为**两供应商复权口径不一致，尚未证明等价**；保留逐日对照，不拼接两源，不声称已消除该差异。数据仅作东方财富后复权收益代理，严格总回报与跨供应商等价性未获证实。来源保留原始/复权双序列及分红窗口，可供复核；这也是本交付的明确限制。

corporate_action_checks.csv列出三股研究期内实际除权除息记录、前后原价与复权价、现金分红/送转算术参照。现金参照仅用于检查事件方向和数量级，不将其与供应商复权公式强行等同。异常收益超过25%的行只进入extreme_return_review.csv（本版{s['off_calendar_records']}条日历外记录，{len(extreme)}条极端收益待审），不裁剪。没有向下游保证复权价可作为成交价。

## 清洗、状态与子集

所有日期先对齐官方日历，再计算相邻交易日收益；收益小数化，不重复除100。000001等代码始终按字符串读取。原始成交量手乘100转股，成交额沿用元；完全重复计数删除，冲突键保留审计并停止其收益使用。

上市日期来自逐股元信息f189；退市历史未独立核实，全部明确unknown。缺行情先排除上市前，再按当日停复牌接口记录确认；无法确认标missing_unknown，不根据无记录或零成交猜停牌。观察到正成交量只记observed_trading，不保证可买卖或排除涨跌停约束。

估计期停牌缺口不补值；剔除条件仅看估计期缺失收益、缺失成交额、冲突和无效数据。检验期有逐日停牌证据、前一日核实估值、且该日无已知公司行动时，延续收盘估值并标不可交易，成交量/额为0；未知或公司行动未能核实时不填。复牌日按前一交易日核实估值计算，可承接整个停牌期间价格变动。长表区分实测价和估值延续。

排名用全部估计交易日平均日成交额，降序，代码升序打破并列；只有合格资产有排名。selection_audit.csv含500只全部理由，model_assets.csv及model_returns_wide.csv按assets中选中资产顺序输出。不会用检验期收益、缺口或涨跌调整名单。100只仅为建议，下游仍须复核协方差与目标估计式。

## 交接边界与未验证项

技术阻塞：{json.dumps(s['technical_blockers'],ensure_ascii=False)}。

公开历史成分接口尝试失败（raw/references/baostock_history_probe.json）；官方已取得的是当前名单，公告日、可得日与生效日留空，不由参考日猜测。降级意味着历史池要求没有严格满足，存在成分前视及存活偏差，报告必须保留这一限定。未独立核实退市历史、供应商公开再分发许可、全量公司行动及严格总回报；组员交叉签认待真实接收人完成，不能把脚本验收当作签认。

建议决策日{periods['decision_date']}，建仓日{periods['proposed_entry_date']}收盘，收益从{periods['proposed_first_return_date']}开始；未执行建仓或任何回测。02任务需要主模型数学检查，03任务需要可交易性与持仓账本，均不属于本次交付。

来源依据与权限见README和数据字典；原始快照raw_inventory.csv列出所有文件SHA256。离线复建与实际验收结果见acceptance.json，只有实际运行后才产生该文件。
'''
    (out/'质量报告.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'data');args=ap.parse_args()
    build(args.output)
