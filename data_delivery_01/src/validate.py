"""Meaningful pipeline acceptance checks, including full offline reconstruction."""
from common import *
from build import build, choose_subset, returns_from_prices, dedup_frame
import argparse, tempfile
from contextlib import contextmanager
import numpy as np
import pandas as pd

@contextmanager
def scratch(prefix):
    # Windows sandbox temp redirection may yield an inaccessible directory.
    # Keep artifacts under the package, and verify the exact cleanup target.
    cache=(ROOT/'.validation_cache').resolve();cache.mkdir(exist_ok=True)
    td=tempfile.TemporaryDirectory(prefix=prefix,dir=cache)
    target=Path(td.name).resolve()
    assert target.is_relative_to(cache) and target!=cache
    try:yield target
    finally:td.cleanup()

def main(data_dir,offline_rebuild=True):
    out=Path(data_dir);checks=[]
    def check(name,condition,detail=''):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
        if not condition:raise AssertionError(name+': '+str(detail))
    c=read_json(ROOT/'snapshot_config.json');m=read_json(out/'manifest.json')
    a=pd.read_csv(out/'assets.csv',dtype={'asset_id':str,'source_code':str,'index_code':str})
    r=pd.read_csv(out/'returns_wide.csv');mask=pd.read_csv(out/'missing_mask.csv')
    p=pd.read_csv(out/'prices_long.csv',dtype={'asset_id':str},low_memory=False)
    ids=a.asset_id.tolist();model=a.loc[a.selected.eq(1),'asset_id'].tolist()
    check('all_500_unique_candidates',len(a)==500 and a.asset_id.nunique()==500)
    check('ticker_leading_zeros_preserved',a.source_code.str.len().eq(6).all() and a.asset_id.str.match(r'^\d{6}\.(SH|SZ)$').all())
    check('identical_labels_order',r.columns.tolist()==['date']+ids and mask.columns.tolist()==r.columns.tolist())
    check('price_unique_keys',not p.duplicated(['date','asset_id']).any())
    check('mask_equals_actual_missingness',np.array_equal(mask[ids].to_numpy(),r[ids].isna().astype(int).to_numpy()))
    check('finite_selected_estimation_returns',np.isfinite(r.loc[r.date.le(c['estimation_end']),model].to_numpy()).all())
    check('subset_count_and_dimension',len(model)==100 and m['counts']['estimation_T']>len(model)+2)
    check('no_unresolved_conflicting_or_invalid_prices',m['counts']['conflicting_date_keys']==0 and m['counts']['invalid_quality_rows']==0)
    check('official_calendar_matches_market_bars',m['counts']['official_calendar_matches_index'])
    for file,info in m['files'].items():check('sha256:'+file,sha(out/file)==info['sha256'])
    for file,digest in m['source_files'].items():check('source_sha256:'+file,sha(ROOT/file)==digest)
    inv=pd.read_csv(out/'raw_inventory.csv')
    check('raw_snapshot_hashes',all(sha(ROOT/row.path)==row.sha256 for row in inv.itertuples()))
    # Hand-calculable gap and price anchors. Inserting a gap must not turn a
    # multi-session price move into a one-session return.
    f=pd.DataFrame({'close_adj':[100.,102.,np.nan,110.,121.],'valuation_valid':[1,1,0,1,1]})
    q=returns_from_prices(f)
    check('hand_calculation_and_gap_propagation',np.isclose(q.iloc[1],.02) and pd.isna(q.iloc[2]) and pd.isna(q.iloc[3]) and np.isclose(q.iloc[4],.1))
    f.loc[2,'close_adj']=102.;f.loc[2,'valuation_valid']=1;q=returns_from_prices(f)
    check('verified_suspension_zero_then_resume',q.iloc[2]==0 and np.isclose(q.iloc[3],110/102-1))
    first=r.date.iloc[0];asset=c['sample_assets'][0]
    stock=p.loc[p.asset_id.eq(asset)].set_index('date')
    expected=stock.at[first,'close_adj']/stock.at[c['anchor_date'],'close_adj']-1
    check('real_first_return_anchor',np.isclose(r.loc[r.date.eq(first),asset].iloc[0],expected,atol=1e-14))
    b=read_json(ROOT/f'raw/prices/{asset}/raw.json')['data']['klines'][0].split(',')
    check('real_volume_lots_to_shares',stock.at[c['anchor_date'],'volume']==float(b[5])*100)
    # Deliberately corrupt every test-period value; selection must not change.
    audit,sel=choose_subset(p,r,ids,c)
    changed_r=r.copy();changed_p=p.copy()
    changed_r.loc[changed_r.date.ge(c['test_start']),ids]=np.nan
    changed_p.loc[changed_p.date.ge(c['test_start']),'amount']=-1e30
    changed_p.loc[changed_p.date.ge(c['test_start']),'quality_invalid']=1
    changed_audit,changed_sel=choose_subset(changed_p,changed_r,ids,c)
    check('test_period_cannot_change_selection',sel==changed_sel and audit.equals(changed_audit))
    # Tie-break must be lexical and independent of input column order.
    tc=dict(c,requested_start='2025-01-02',estimation_end='2025-01-02',subset_size=1)
    tp=pd.DataFrame({'date':['2025-01-02']*2,'asset_id':['000002.SZ','000001.SZ'],'amount':[100.,100.],
                     'key_conflict':[0,0],'quality_invalid':[0,0]})
    tr=pd.DataFrame({'date':['2025-01-02'],'000002.SZ':[.01],'000001.SZ':[.02]})
    _,ts=choose_subset(tp,tr,['000002.SZ','000001.SZ'],tc)
    check('liquidity_tie_break_by_code',ts==['000001.SZ'])
    # Exact duplicate and conflicting duplicate handling are different outcomes.
    with scratch('synthetic_') as td:
        path=Path(td)/'duplicate.json';base='2025-01-02,1,1,1,1,10,100,0,0,0,1'
        write_json(path,{'data':{'klines':[base,base,base.replace(',1,1,1,1,',',1,2,2,1,')]}})
        d=[];result=dedup_frame(path,'raw','000001.SZ',d)
        check('exact_duplicates_counted_conflicts_flagged',d[0]['exact_duplicates']==1 and d[0]['conflicting_dates']==1 and result.key_conflict.iloc[0])
    if offline_rebuild:
        # This rebuild only reads ROOT/raw and snapshot_config. No acquisition
        # functions are called, and an isolated directory receives all results.
        with scratch('offline_rebuild_') as td:
            rebuilt=Path(td);build(rebuilt)
            hashes={name:sha(rebuilt/name) for name in m['files']}
            same=all(digest==m['files'][name]['sha256'] for name,digest in hashes.items())
            check('complete_offline_rebuild_identical_file_hashes',same)
            check('offline_manifest_identical',sha(rebuilt/'manifest.json')==sha(out/'manifest.json'))
    result={'data_version':m['data_version'],'checks_passed':len(checks),'checks_failed':0,
        'offline_rebuild_executed':offline_rebuild,'checks':checks,
        'technical_blockers':m['technical_blockers'],'limitations':m['limitations'],
        'human_cross_member_signoff':'pending_not_simulated'}
    write_json(out/'acceptance.json',result)
    print('ACCEPTANCE PASSED',len(checks),'checks; technical blockers:',m['technical_blockers'],flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'data')
    ap.add_argument('--skip-rebuild',action='store_true');args=ap.parse_args()
    main(args.data,not args.skip_rebuild)
