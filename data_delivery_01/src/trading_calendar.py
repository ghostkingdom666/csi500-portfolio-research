"""Exchange calendar reconstructed from the frozen official SSE holiday notices."""
from common import ROOT, read_json
import pandas as pd

HOLIDAYS={
 2025:[('01-01','01-01'),('01-28','02-04'),('04-04','04-06'),('05-01','05-05'),('05-31','06-02'),('10-01','10-08')],
 2026:[('01-01','01-03'),('02-15','02-23'),('04-04','04-06'),('05-01','05-05'),('06-19','06-21'),('09-25','09-27'),('10-01','10-07')],
}

def calendar(c):
    # Files are mandatory evidence; no fallback to a generic weekday calendar.
    for year in HOLIDAYS:
        assert (ROOT/f'raw/references/sse_{year}.html').exists(), f'missing official {year} notice'
    holidays=set()
    for year, intervals in HOLIDAYS.items():
        for start,end in intervals:
            holidays.update(pd.date_range(f'{year}-{start}',f'{year}-{end}').strftime('%Y-%m-%d'))
    ds=pd.date_range(c['anchor_date'],c['closed_date_candidate'])
    frame=pd.DataFrame({'date':ds.strftime('%Y-%m-%d')})
    frame['is_open']=[int(d.weekday()<5 and s not in holidays) for d,s in zip(ds,frame.date)]
    frame['is_anchor']=(frame.date<c['requested_start']).astype(int)
    frame['period']=['anchor' if d<c['requested_start'] else 'estimation' if d<=c['estimation_end'] else 'test' for d in frame.date]
    frame['source']=['SSE official annual holiday notice '+s[:4] if s[:4]!='2024' else '2024-12-31 anchor corroborated by CSI500 index bars' for s in frame.date]
    return frame
