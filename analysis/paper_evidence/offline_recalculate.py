"""Offline verification of the original paper's printed 14-stock portfolio.

Run: python offline_recalculate.py
Requires numpy. Reads only files next to __file__, never calls the network and
never accesses the original project or a database. The whole directory may move.
Inputs: paper-14-coefficients.json, request-manifest.json, 28 tencent-*.json files.
Outputs: offline-summary.json, offline-*-returns.csv and offline-checks.json.

Original p9/physical p9 Table2 prints raw coefficients, not invested weights.
The original p8/original p24 method normalizes them and applies the same weights
to each daily return, i.e. daily constant-weight rebalancing. A separate buy/hold
calculation isolates that accounting choice. Raw close pct changes and Tencent
hfq close pct changes are alternative vendor definitions, not identical JY data.
"""
from pathlib import Path
import csv, hashlib, json
from decimal import Decimal
import numpy as np

ROOT = Path(__file__).resolve().parent

def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def write_json(name, data):
    (ROOT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def write_csv(name, rows):
    with (ROOT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def metrics(nav):
    nav = np.asarray(nav, float)
    r = nav[1:]/nav[:-1]-1
    assert len(r) == 21
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    cum = float(nav[-1]/nav[0]-1)
    rf_daily = 1.015**(1/252)-1
    rf_monthly = 1.015**(1/12)-1
    return dict(n_return_days=len(r), cumulative_return=cum,
        daily_mean=mean, daily_sd=sd,
        annualized_geometric_252=(1+cum)**(252/len(r))-1,
        annualized_monthly_compound_12=(1+cum)**12-1,
        paper_mixed_frequency_sharpe=(mean-rf_monthly)/sd,
        annualized_sharpe_rf_1_5pct=(mean-rf_daily)/sd*np.sqrt(252),
        annualized_sharpe_rf_0=mean/sd*np.sqrt(252),
        max_drawdown=float((nav/np.maximum.accumulate(nav)-1).min()))

def main():
    coeff = load('paper-14-coefficients.json')['holdings']
    assert len(coeff) == 14
    ids = [r['asset_id'] for r in coeff]
    c = np.array([r['raw_coefficient'] for r in coeff], float)
    u = c/c.sum()
    assert len(set(ids)) == 14 and np.all(c > 0)
    assert abs(c.sum()-5.3547) < 1e-12 and abs(u.sum()-1) < 1e-12
    manifest = load('request-manifest.json')
    assert len(manifest) == 28 and all(r['status']=='ok' for r in manifest)
    sha_ok = all(hashlib.sha256((ROOT/r['response_file']).read_bytes()).hexdigest()==r['sha256'] for r in manifest)
    assert sha_ok, 'A retained response differs from its original request manifest.'
    bank = {'601077.SH','601229.SH','601288.SH'}
    total_decimal = sum(Decimal(str(r['raw_coefficient'])) for r in coeff)
    bank_decimal = sum(Decimal(str(r['raw_coefficient'])) for r in coeff if r['asset_id'] in bank)
    top4_decimal = sum(sorted((Decimal(str(r['raw_coefficient'])) for r in coeff), reverse=True)[:4])
    result = dict(
        scope='Printed 14-stock portfolio on public Tencent prices, not original 278-stock estimation reconstruction',
        source='Retained Tencent fqkline responses; all 28 response hashes checked',
        anchor='2025-03-31 close', return_start='2025-04-01', return_end='2025-04-30',
        n_return_days=21, raw_coefficient_sum=float(c.sum()),
        original_paper_claims=dict(cumulative_return=.0551,daily_mean=.00265,daily_sd=.0137,sharpe=.1023,annualized_return=.9034),
        annualization_arithmetic=dict(monthly_5_51_compounded_12=1.0551**12-1,
            note='One-month extrapolation; 252/21=12, not a one-year realized return.'),
        allocation=dict(bank_codes=sorted(bank),bank_raw_coefficient_sum=str(bank_decimal),
            bank_normalized_weight_exact_ratio=f'{bank_decimal}/{total_decimal}',bank_normalized_weight=float(bank_decimal/total_decimal),
            top4_assets=[ids[i] for i in np.argsort(-u)[:4]],top4_raw_coefficient_sum=str(top4_decimal),
            top4_normalized_weight_exact_ratio=f'{top4_decimal}/{total_decimal}',top4_normalized_weight=float(top4_decimal/total_decimal),
            max_weight=float(u.max()),hhi=float(u@u),effective_holdings=float(1/(u@u)),
            note='Bank category includes only 601077/601229/601288; 605499 is Dongpeng Beverage.'),
        normalized_holdings=[dict(asset_id=asset,raw_coefficient=float(c[i]),invested_weight=float(u[i])) for i,asset in enumerate(ids)],
        limitation='Original full-precision coefficients and JY return panel unavailable. A mismatch cannot identify its original cause or establish misconduct.')
    checks = dict(response_count=28,response_hashes_match=sha_ok,positive_coefficients=True,normalized_sum=float(u.sum()))
    for kind in ['raw','hfq']:
        close_maps=[]
        for holding in coeff:
            symbol=holding['exchange'].lower()+holding['code']
            response=load(f'tencent-{symbol}-{kind}.json')
            assert response['code']==0
            data=response['data'][symbol]
            rows=data['hfqday'] if kind=='hfq' and 'hfqday' in data else data['day']
            filtered=[row for row in rows if '2025-03-31' <= row[0] <= '2025-04-30']
            assert len(filtered)==22 and len({row[0] for row in filtered})==22
            close_maps.append({row[0]:float(row[2]) for row in filtered})
        dates=sorted(close_maps[0])
        assert dates[0]=='2025-03-31' and dates[-1]=='2025-04-30' and dates[1]=='2025-04-01'
        assert all(sorted(m)==dates for m in close_maps)
        P=np.array([[m[d] for m in close_maps] for d in dates],float)
        assert np.isfinite(P).all() and np.all(P>0)
        R=P[1:]/P[:-1]-1
        fixed_returns=R@u
        rebalanced=np.r_[1.,np.cumprod(1+fixed_returns)]
        buyhold=(P/P[0])@u
        index=ids.index('601127.SH')
        # Single asset's daily rebalancing is identical to buying and holding it.
        benchmark=P[:,index]/P[0,index]
        benchmark_rebalanced=np.r_[1.,np.cumprod(1+R[:,index])]
        assert np.max(abs(benchmark-benchmark_rebalanced)) < 1e-12
        assert abs(buyhold[-1]-(1+np.sum(u*(P[-1]/P[0]-1)))) < 1e-12
        # Enumerate every coefficient-box vertex. Each day's portfolio return is
        # a linear-fractional function of positive coefficients, so its extrema
        # over that box occur at vertices. Multiplying independent daily min/max
        # bounds yields conservative bounds for ALL coefficient rounding choices.
        half=.00005
        lower=np.maximum(c-half,0);upper=c+half
        bits=(np.arange(2**len(c))[:,None] >> np.arange(len(c))) & 1
        vertices=lower+(upper-lower)*bits
        vertices/=vertices.sum(axis=1,keepdims=True)
        vertex_daily=R@vertices.T
        vertex_cumulative=np.prod(1+vertex_daily,axis=0)-1
        rounding=dict(decimal_half_width=half,
            conservative_cumulative_min=float(np.prod(1+vertex_daily.min(axis=1))-1),
            conservative_cumulative_max=float(np.prod(1+vertex_daily.max(axis=1))-1),
            vertex_cumulative_min=float(vertex_cumulative.min()),vertex_cumulative_max=float(vertex_cumulative.max()),
            note='Conservative min/max cover any coefficients within half a last printed decimal; vertex cumulative extrema alone are diagnostic, not guaranteed interior extrema.')
        paper_value=.0551
        rounding['paper_5_51_inside_conservative_bounds']=bool(rounding['conservative_cumulative_min']<=paper_value<=rounding['conservative_cumulative_max'])
        m=metrics(rebalanced)
        result[kind]=dict(main14_daily_rebalanced=m,main14_buy_and_hold=metrics(buyhold),
            benchmark601127_daily_rebalanced=metrics(benchmark_rebalanced),
            benchmark601127_buy_and_hold=metrics(benchmark),
            benchmark601127_sharpe_stale_main_paper_mean=(.00265-(1.015**(1/12)-1))/R[:,index].std(ddof=1),
            gap_from_paper_bp=(paper_value-m['cumulative_return'])*10000,
            weight_rounding_bounds=rounding,
            per_asset=[dict(asset_id=ids[i],cumulative_return=float(P[-1,i]/P[0,i]-1),daily_mean=float(R[:,i].mean()),daily_sd=float(R[:,i].std(ddof=1))) for i in range(len(ids))])
        write_csv(f'offline-{kind}-returns.csv',[
            dict(date=d,**{asset:float(R[i,j]) for j,asset in enumerate(ids)},
                main14_daily_rebalanced_return=float(fixed_returns[i]),main14_daily_rebalanced_nav=float(rebalanced[i+1]),
                main14_buy_and_hold_return=float(buyhold[i+1]/buyhold[i]-1),main14_buy_and_hold_nav=float(buyhold[i+1]),
                benchmark601127_return=float(R[i,index]),benchmark601127_nav=float(benchmark[i+1]))
            for i,d in enumerate(dates[1:])])
        checks[kind]=dict(anchor_and21_identical_return_dates=True,positive_finite_closes=True,
            benchmark_rebalanced_buyhold_equal=True,buyhold_endpoint_matches_weighted_endpoint=True,
            original_claim_outside_print_rounding_bounds=not rounding['paper_5_51_inside_conservative_bounds'])
    write_json('offline-summary.json',result)
    write_json('offline-checks.json',checks)
    print(json.dumps(dict(raw=result['raw']['main14_daily_rebalanced'],hfq=result['hfq']['main14_daily_rebalanced'],allocation=result['allocation'],checks=checks),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
