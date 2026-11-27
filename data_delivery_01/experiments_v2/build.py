"""Build the wider CSI 500 panels from the immutable 01 data delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'data'
VERSION = 'csi500_rolling_v2'
SIZES = (100, 200, 250, 300)
WINDOWS = (
    ('2026-05_06', '2026-05-01', '2026-06-30', 'main'),
    ('2026-07_08', '2026-07-01', '2026-08-31', 'main'),
    ('2026-09_10', '2026-09-01', '2026-10-31', 'main'),
    ('2026-11_partial', '2026-11-01', None, 'short_supplement'),
    ('2026-07_original', '2026-07-01', None, 'original_comparison'),
)
INPUTS = ('assets.csv', 'returns_wide.csv', 'prices_long.csv', 'index_daily.csv',
          'calendar.csv', 'periods.json', 'model_assets.csv')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_out(frame, path, index=False):
    frame.to_csv(path, index=index, encoding='utf-8-sig', lineterminator='\n',
                 float_format='%.17g')


def read_snapshot():
    manifest_path = SOURCE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashes = {name: sha256(SOURCE / name) for name in INPUTS}
    for name, digest in hashes.items():
        require(digest == manifest['files'][name]['sha256'], f'Input hash mismatch: {name}')
    hashes['manifest.json'] = sha256(manifest_path)

    assets = pd.read_csv(SOURCE / 'assets.csv', dtype=str).set_index('asset_id')
    returns = pd.read_csv(SOURCE / 'returns_wide.csv', index_col='date', float_precision='round_trip')
    prices = pd.read_csv(
        SOURCE / 'prices_long.csv', float_precision='round_trip',
        usecols=['date', 'asset_id', 'amount', 'close_raw', 'close_adj', 'tradable_status',
                 'status_source', 'valuation_method', 'key_conflict', 'quality_invalid'],
    ).set_index(['date', 'asset_id']).sort_index()
    index_close = pd.read_csv(SOURCE / 'index_daily.csv', index_col='date',
                              float_precision='round_trip')['close']
    calendar = pd.read_csv(SOURCE / 'calendar.csv', dtype={'date': str})
    research_dates = calendar.loc[calendar['is_anchor'].eq(0), 'date'].tolist()
    require(len(assets) == 500 and assets.index.is_unique and
            assets.index.tolist() == returns.columns.tolist(), 'Unexpected candidate labels')
    require(returns.index.is_unique and returns.index.is_monotonic_increasing and
            returns.index.tolist() == research_dates, 'Return dates differ from trading calendar')
    require(prices.index.is_unique, 'Duplicate daily price key')
    require(index_close.reindex(returns.index).gt(0).all(), 'Missing or invalid index close')
    require(returns.index[-1] == manifest['last_complete_trading_date'], 'Unexpected snapshot cutoff')
    return manifest, hashes, assets, returns, prices, index_close


def select_assets(estimate, estimate_prices):
    """Use only data that the caller has sliced to the estimation window."""
    ids = estimate.columns
    T = len(estimate)
    finite_returns = np.isfinite(estimate).sum()
    valid_amount = np.isfinite(estimate_prices['amount']) & estimate_prices['amount'].ge(0)
    amount_days = valid_amount.groupby(level='asset_id').sum().reindex(ids, fill_value=0)
    amount_mean = estimate_prices['amount'].where(valid_amount).groupby(level='asset_id').mean().reindex(ids)
    issue = estimate_prices['key_conflict'].ne(0) | estimate_prices['quality_invalid'].ne(0)
    invalid_days = issue.groupby(level='asset_id').sum().reindex(ids, fill_value=0)
    audit = pd.DataFrame({
        'finite_estimation_returns': finite_returns,
        'valid_estimation_amount_days': amount_days,
        'estimation_issue_days': invalid_days,
        'mean_daily_amount_cny': amount_mean.where(amount_days.eq(T)),
    })
    audit.index.name = 'asset_id'
    audit['eligible'] = finite_returns.eq(T) & amount_days.eq(T) & invalid_days.eq(0)
    order = audit.loc[audit['eligible']].reset_index().sort_values(
        ['mean_daily_amount_cny', 'asset_id'], ascending=[False, True])['asset_id'].tolist()
    audit['rank'] = pd.Series(range(1, len(order) + 1), index=order, dtype='Int64')
    return audit, order


def matrix_diagnostic(estimate):
    values = estimate.to_numpy(dtype=float)
    T, N = values.shape
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = singular[0] * max(T, N) * np.finfo(values.dtype).eps
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        'estimation_T': T, 'asset_count_N': N, 'T_minus_N_minus_2': T - N - 2,
        'T_over_N': round(T / N, 6), 'T_gt_N_plus_2': T > N + 2,
        'centered_return_rank': rank, 'sample_covariance_full_rank': rank == N,
        'sample_covariance_condition_number': round(float((singular[0] / singular[-1]) ** 2), 5)
        if rank == N else None,
    }


def build_window(out, spec, assets, returns, prices, index_close):
    window_id, start, end, role = spec
    test = returns.loc[start:end]
    require(len(test) >= 2, f'{window_id}: fewer than two test sessions')
    estimate = returns.loc[returns.index < test.index[0]]
    est_prices = prices.loc[pd.IndexSlice[:estimate.index[-1], :], :]
    est_prices = est_prices.loc[est_prices.index.get_level_values('date') >= estimate.index[0]]
    audit, rank = select_assets(estimate, est_prices)
    require(len(rank) >= max(SIZES), f'{window_id}: fewer than 300 eligible stocks')
    require(len(estimate) > max(SIZES) + 2, f'{window_id}: T must exceed N + 2')

    selected = sorted(rank[:max(SIZES)])
    entry = test.index[0]
    pnl = test.iloc[1:]
    entry_prices = prices.xs(entry, level='date').reindex(audit.index)
    entry_trade_bar = (entry_prices['tradable_status'].eq('observed_trading') &
                       entry_prices['valuation_method'].eq('vendor_close') &
                       np.isfinite(entry_prices['close_raw']) & entry_prices['close_raw'].gt(0))
    test_status = prices.loc[pd.IndexSlice[entry:test.index[-1], selected], :]
    test_carry = test_status['valuation_method'].eq('verified_suspension_carry')
    audit['name'] = assets['name']
    audit['listing_date'] = assets['listing_date']
    audit['entry_trade_bar_observed'] = entry_trade_bar
    audit['entry_tradable_status'] = entry_prices['tradable_status']
    audit['entry_close_raw'] = entry_prices['close_raw']
    audit['entry_close_adj'] = entry_prices['close_adj']
    audit['holding_missing_returns'] = (~np.isfinite(pnl)).sum()

    benchmark = index_close.reindex(test.index).rename('index_close').to_frame()
    benchmark['holding_return'] = benchmark['index_close'].pct_change(fill_method=None)
    benchmark['index_nav'] = benchmark['index_close'] / benchmark['index_close'].iloc[0]
    index_return = float(benchmark['index_nav'].iloc[-1] - 1)
    regime = 'up' if index_return >= 0.05 else 'down' if index_return <= -0.05 else 'mixed'
    common = {
        'window_id': window_id, 'role': role, 'scheduled_start': start,
        'scheduled_end': end, 'estimation_start': estimate.index[0],
        'decision_date': estimate.index[-1], 'entry_date': entry,
        'first_holding_return_date': pnl.index[0], 'test_end': pnl.index[-1],
        'holding_sessions': len(pnl), 'eligible_count': len(rank),
        'csi500_price_index_return': round(index_return, 10), 'realized_regime': regime,
    }
    results = []
    for N in SIZES:
        ids = sorted(rank[:N])
        audit[f'in_{N}'] = audit.index.isin(ids)
        diag = matrix_diagnostic(estimate[ids])
        entry_gaps = int((~entry_trade_bar.loc[ids]).sum())
        results.append({**common, **diag,
                        'entry_assets_without_observed_trading': entry_gaps,
                        'entry_status': 'requires_common_execution_rule' if entry_gaps else 'observed_bars_for_all_assets',
                        'missing_holding_return_cells': int((~np.isfinite(pnl[ids])).sum().sum()),
                        'suspension_valuation_rows': int(test_carry.loc[pd.IndexSlice[:, ids]].sum())})

    directory = out / window_id
    directory.mkdir()
    csv_out(estimate[selected], directory / 'estimation_returns.csv', index=True)
    csv_out(pnl[selected], directory / 'holding_returns.csv', index=True)
    csv_out(benchmark, directory / 'benchmark.csv', index=True)
    csv_out(audit, directory / 'assets.csv', index=True)
    csv_out(test_status.loc[~test_status['tradable_status'].eq('observed_trading'),
                            ['tradable_status', 'valuation_method', 'status_source']],
            directory / 'test_status_exceptions.csv', index=True)
    return results, rank


def build(output):
    manifest, input_hashes, assets, returns, prices, index_close = read_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    old_periods = json.loads((SOURCE / 'periods.json').read_text(encoding='utf-8'))
    for spec in WINDOWS:
        rows, rank = build_window(output, spec, assets, returns, prices, index_close)
        if spec[-1] == 'original_comparison':
            old = pd.read_csv(SOURCE / 'model_assets.csv')['asset_id'].tolist()
            require(sorted(rank[:100]) == old, 'Original 100-stock baseline differs from frozen delivery')
            require(rows[0]['decision_date'] == old_periods['decision_date'] and
                    rows[0]['entry_date'] == old_periods['proposed_entry_date'] and
                    rows[0]['test_end'] == old_periods['test_end'], 'Original comparison dates changed')
        summary.extend(rows)
    csv_out(pd.DataFrame(summary), output / 'windows.csv')
    outputs = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*.csv'))}
    metadata = {
        'version': VERSION, 'source_version': manifest['data_version'],
        'source_manifest_sha256': input_hashes['manifest.json'], 'input_sha256': input_hashes,
        'output_sha256': outputs, 'build_sha256': sha256(Path(__file__)),
        'pool_type': manifest['pool_type'], 'stock_counts': list(SIZES),
        'market_regime_threshold': 0.05, 'index_return_type': 'price_index',
        'return_type': 'supplier_adjusted_price_return_proxy', 'windows': summary,
    }
    (output / 'manifest.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'data',
                        help='New output directory (must not already exist)')
    build(parser.parse_args().output)
