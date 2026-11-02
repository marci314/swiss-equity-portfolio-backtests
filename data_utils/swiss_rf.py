from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Loading / building the daily risk-free series
# --------------------------------------------------------------------------

def load_swiss_rf_monthly(csv_path: str) -> pd.Series:
    """Load the monthly Swiss call-money rate as a *decimal* annual rate.

    Returns a Series indexed by month-start dates, e.g. 0.0163 for 1.63 %/yr.
    """
    df = pd.read_csv(csv_path)
    col = [c for c in df.columns if c != 'observation_date'][0]
    s = (
        df.assign(observation_date=pd.to_datetime(df['observation_date']))
          .set_index('observation_date')[col]
          .sort_index()
          .astype(float)
    )
    s = s / 100.0
    s.name = 'rf_annual'
    return s


def build_daily_rf(
    csv_path: str,
    trading_index: pd.DatetimeIndex,
    day_count: int = 360,
) -> pd.Series:
    """Build a *daily* risk-free return series aligned to `trading_index`.

    The annualised rate that applies over a holding period is the rate known
    at the *start* of that period (we therefore forward-fill the monthly rate
    and shift it by one observation, exactly like
    `RfDaily = AnnualRate.shift() * DayCount / 360`).

    Parameters
    ----------
    csv_path      : path to IRSTCI01CHM156N.csv
    trading_index : the DatetimeIndex of the (daily) return series to align to
    day_count     : 360 for money-market ACT/360 (SNB convention)

    Returns
    -------
    pd.Series of per-day risk-free returns (decimal), indexed by trading_index.
    The first observation is 0 (no preceding day to accrue over).
    """
    rf_annual = load_swiss_rf_monthly(csv_path)
    trading_index = pd.DatetimeIndex(trading_index).sort_values()

    # Forward-fill the monthly annual rate onto every trading day, then use the
    # rate prevailing at the *previous* day to accrue interest over the gap.
    rf_on_days = rf_annual.reindex(
        rf_annual.index.union(trading_index)
    ).ffill().reindex(trading_index)

    # If the backtest starts before the first available rate, back-fill so we
    # do not silently inject zeros (uses the earliest known rate).
    rf_on_days = rf_on_days.bfill()

    day_count_days = (
        trading_index.to_series().diff().dt.days.fillna(0).to_numpy()
    )
    prev_rate = rf_on_days.shift(1).fillna(rf_on_days.iloc[0]).to_numpy()
    rf_daily = prev_rate * day_count_days / day_count

    return pd.Series(rf_daily, index=trading_index, name='rf_daily')


# --------------------------------------------------------------------------
# Cash overlay for partially-invested / trend-timed strategies
# --------------------------------------------------------------------------

def apply_cash_overlay(
    strategy_returns: pd.Series,
    exposure: pd.Series,
    rf_daily: pd.Series,
) -> pd.Series:
    """Combine a (0..1) market-exposure overlay with the risk-free rate.

    r_p = exposure * r_strategy + (1 - exposure) * rf

    This is the correct treatment of risk-off capital: when a trend filter
    pulls the book to cash (exposure -> 0) it should earn the risk-free rate,
    not 0. Replaces a plain `exposure * r_raw` calculation.
    """
    idx = strategy_returns.index
    expo = exposure.reindex(idx).fillna(0.0)
    rf = rf_daily.reindex(idx).fillna(0.0)
    return expo * strategy_returns + (1.0 - expo) * rf


def to_excess(returns: pd.Series, rf_daily: pd.Series) -> pd.Series:
    """Excess return = total return - aligned daily risk-free rate."""
    return returns - rf_daily.reindex(returns.index).fillna(0.0)


# --------------------------------------------------------------------------
# RF-aware performance summary
# --------------------------------------------------------------------------

def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


def performance_summary(
    returns: pd.Series,
    rf_daily: pd.Series,
    bm: Optional[pd.Series] = None,
    n: int = 252,
) -> pd.Series:
    """Full RF-aware performance statistics for one daily return series.

    All risk-adjusted figures use EXCESS returns over the Swiss risk-free rate.
    Alpha / beta / tracking error / information ratio are estimated against the
    benchmark's *excess* return (CAPM-style).
    """
    r = returns.dropna()
    rf = rf_daily.reindex(r.index).fillna(0.0)
    xs = r - rf                                    # excess returns

    cagr = (1 + r).prod() ** (n / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(n)
    xs_vol = xs.std(ddof=1) * np.sqrt(n)
    sharpe = (xs.mean() * n) / xs_vol if xs_vol > 0 else np.nan
    geo_xs = (1 + xs).prod() ** (n / len(xs)) - 1   # geometric excess
    mdd = _max_drawdown(r)

    out = {
        'CAGR': cagr,
        'Vol': vol,
        'Sharpe': sharpe,                # excess-return Sharpe
        'GeoExcess': geo_xs,
        'MaxDD': mdd,
        'MeanRF_ann': rf.mean() * n,     # sanity check: avg rf actually applied
    }

    if bm is not None:
        b = bm.reindex(r.index)
        bx = (b - rf).dropna()           # benchmark excess return
        xs_a = xs.reindex(bx.index)
        # CAPM regression of strategy excess on benchmark excess
        cov = np.cov(xs_a, bx, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        alpha_daily = xs_a.mean() - beta * bx.mean()
        out['Alpha_ann'] = alpha_daily * n
        out['Beta'] = beta
        active = (r - b).dropna()        # active return vs benchmark (total)
        te = active.std(ddof=1) * np.sqrt(n)
        out['TrackingError'] = te
        out['IR_vs_SPI'] = (active.mean() * n) / te if te > 0 else np.nan

    return pd.Series(out)


def performance_table(
    returns_dict: dict[str, pd.Series],
    rf_daily: pd.Series,
    bm_key: str = 'SPI',
    n: int = 252,
    round_pct: bool = True,
) -> pd.DataFrame:
    """Build a comparison table for several return series at once.

    `returns_dict` maps a label -> daily return series; the series under
    `bm_key` is used as the benchmark for the others.
    """
    bm = returns_dict.get(bm_key)
    rows = {}
    for label, r in returns_dict.items():
        use_bm = None if label == bm_key else bm
        rows[label] = performance_summary(r, rf_daily, bm=use_bm, n=n)
    tbl = pd.DataFrame(rows).T
    if round_pct:
        for c in ['CAGR', 'Vol', 'GeoExcess', 'MaxDD', 'Alpha_ann',
                  'TrackingError', 'MeanRF_ann']:
            if c in tbl:
                tbl[c] = (tbl[c] * 100).round(2)
        for c in ['Sharpe', 'IR_vs_SPI', 'Beta']:
            if c in tbl:
                tbl[c] = tbl[c].round(2)
    return tbl
