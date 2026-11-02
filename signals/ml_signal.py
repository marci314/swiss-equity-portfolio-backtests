# --------------------------------------------------------------------------
# model_type options
#   'ridge'         : Ridge regression (L2)                           [regression]
#   'enet'          : ElasticNet regression (L1 + L2)                 [regression]
#   'xgb'           : XGBRegressor, gradient-boosted trees            [regression]
#   'ranker_ndcg'   : XGBRanker, rank:ndcg, EXPONENTIAL gain 2^rel-1  [learning-to-rank]
#                     -> the canonical NDCG formulation
#   'lambdamart'    : XGBRanker, rank:ndcg, LINEAR gain (rel)         [learning-to-rank]
#                     -> LambdaMART with linear NDCG gain (ndcg_exp_gain=False)
#   'ranker'        : alias of 'ranker_ndcg' (backward compatibility)
# --------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

# Model types that are learning-to-rank (need integer relevance labels + groups)
RANKER_TYPES = ('ranker', 'ranker_ndcg', 'lambdamart')

# Default JKP feature set
DEFAULT_FEATURES = [
    'ret_6_1',      # Momentum 6-1
    'ret_12_1',     # Momentum 12-1
    'qmj',          # Quality minus Junk
    'qmj_growth',   # Growth component
    'qmj_safety',   # Safety component
    'gp_at',        # Gross profitability
    'op_at',        # Operating profitability
    'be_me',        # Book-to-market (value)
    'debt_me',      # Leverage
    'at_gr1',       # Asset growth
    'oaccruals_at', # Accruals
]


# --------------------------------------------------------------------------
# Cross-sectional transforms (winsorise -> z-score -> clip)
# --------------------------------------------------------------------------

def cs_standardize(df: pd.DataFrame,
                   lower: float = 0.01,
                   upper: float = 0.99,
                   clip: float = 3.0) -> pd.DataFrame:
    """Winsorise then z-score each column cross-sectionally; fill gaps with 0."""
    X = df.apply(pd.to_numeric, errors='coerce').astype(np.float64)
    lo, hi = X.quantile(lower), X.quantile(upper)
    X = X.clip(lower=lo, upper=hi, axis=1)
    X = (X - X.mean()) / X.std(ddof=0).replace(0.0, np.nan)
    return X.clip(-clip, clip).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def cs_zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)


# --------------------------------------------------------------------------
# Point-in-time feature / label extraction
# --------------------------------------------------------------------------

def jkp_asof(bs, date: str, fields: list, lag_days: int = 90) -> pd.DataFrame:
    """Last available JKP characteristics per id, as of `date - lag_days`.

    The lag respects the fact that fundamentals are only known with a
    reporting delay, preventing forward-looking bias.
    """
    asof = pd.to_datetime(date) - pd.Timedelta(days=lag_days)
    df = bs.data.jkp_data[fields]
    df = df.loc[df.index.get_level_values('date') <= asof]
    if df.empty:
        return pd.DataFrame(columns=fields)
    return df.groupby(level='id').last()


def fwd_return(bs, d0: str, d1: str, ids) -> pd.Series:
    """Realised cross-sectional total return between two rebalancing dates."""
    r = bs.data.get_return_series(
        end_date=d1, width=800, weekdays_only=True, fillna_value=None)
    r = r.loc[(r.index > pd.to_datetime(d0)) & (r.index <= pd.to_datetime(d1))]
    return ((1 + r.reindex(columns=ids)).prod(axis=0) - 1).dropna()


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------

def _build_model(model_type: str, **kw):
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if model_type == 'ridge':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=kw.get('ridge_alpha', 10.0))),
        ])

    if model_type == 'enet':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('enet', ElasticNet(
                alpha=kw.get('enet_alpha', 0.01),     # overall penalty strength
                l1_ratio=kw.get('l1_ratio', 0.5),     # 0 = ridge, 1 = lasso, 0.5 = mix
                max_iter=kw.get('enet_max_iter', 5000),
            )),
        ])

    if model_type == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective='reg:squarederror',
            max_depth=kw.get('max_depth', 4),
            learning_rate=kw.get('eta', 0.05),
            n_estimators=kw.get('n_estimators', 200),
            subsample=kw.get('subsample', 0.8),
            colsample_bytree=kw.get('colsample_bytree', 0.8),
            reg_lambda=kw.get('reg_lambda', 1.0),
            n_jobs=kw.get('n_jobs', -1),
        )

    if model_type in ('ranker', 'ranker_ndcg'):
        # Canonical NDCG: gain = 2^rel - 1  (ndcg_exp_gain=True)
        from xgboost import XGBRanker
        return XGBRanker(
            objective='rank:ndcg',
            ndcg_exp_gain=kw.get('ndcg_exp_gain', True),
            max_depth=kw.get('max_depth', 4),
            learning_rate=kw.get('eta', 0.05),
            n_estimators=kw.get('n_estimators', 200),
            subsample=kw.get('subsample', 0.8),
            colsample_bytree=kw.get('colsample_bytree', 0.8),
            reg_lambda=kw.get('reg_lambda', 1.0),
            min_child_weight=kw.get('min_child_weight', 5),
            n_jobs=kw.get('n_jobs', -1),
            random_state=0,
        )

    if model_type == 'lambdamart':
        # LambdaMART ranker: rank:ndcg with LINEAR gain (ndcg_exp_gain=False).
        from xgboost import XGBRanker
        return XGBRanker(
            objective='rank:ndcg',
            ndcg_exp_gain=kw.get('ndcg_exp_gain', False),   # linear NDCG gain
            n_estimators=kw.get('n_estimators', 100),
            max_depth=kw.get('max_depth', 6),
            learning_rate=kw.get('learning_rate', 0.1),
            min_child_weight=kw.get('min_child_weight', 1),
            subsample=kw.get('subsample', 0.8),
            n_jobs=kw.get('n_jobs', -1),
            random_state=0,
        )

    raise ValueError(f"Unknown model_type '{model_type}'")


# --------------------------------------------------------------------------
# The backtest item-builder
# --------------------------------------------------------------------------

def bibfn_ml_signal(bs, rebdate: str, **kwargs) -> None:
    """Generate ML-based cross-sectional scores and store them on bs.

    kwargs
    ------
    fields            : list of JKP feature columns      (default DEFAULT_FEATURES)
    model_type        : see module header                (default 'ridge')
    train_window_obs  : number of past rebalances used for training (default 36)
    lag_days          : reporting lag for fundamentals   (default 90)
    retrain_every     : refit cadence in rebalances      (default 1)
    rank_label        : train on cross-sectional return *ranks* (default False;
                        forced True for any ranker)
    field_name        : column name written to scores    (default 'scores')
    Any extra kwargs (ridge_alpha, enet_alpha, l1_ratio, max_depth, eta,
    ndcg_exp_gain, ...) are forwarded to the model.
    """
    fields     = kwargs.get('fields', DEFAULT_FEATURES)
    model_type = kwargs.get('model_type', 'ridge')
    n_train    = kwargs.get('train_window_obs', 36)
    lag_days   = kwargs.get('lag_days', 90)
    retrain    = max(1, int(kwargs.get('retrain_every', 1)))
    is_ranker  = model_type in RANKER_TYPES
    rank_label = kwargs.get('rank_label', False) or is_ranker
    field_name = kwargs.get('field_name', 'scores')
    rebdates   = bs.settings['rebdates']
    ids        = list(bs.selection.selected)

    # Hyper-parameters forwarded to the model (exclude control keys)
    _control = {'fields', 'model_type', 'train_window_obs', 'lag_days',
                'retrain_every', 'rank_label', 'field_name', 'item_name'}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in _control}

    def _empty():
        bs.optimization_data['scores'] = pd.DataFrame(
            {field_name: pd.Series(0.0, index=ids, dtype=np.float64)})

    # Keep only features that actually exist in the data
    avail = set(bs.data.jkp_data.columns)
    fields = [f for f in fields if f in avail]
    if not fields or len(ids) == 0:
        _empty()
        return None

    # ---- current-date features (the prediction input) ----
    Xnow = cs_standardize(jkp_asof(bs, rebdate, fields, lag_days).reindex(ids))

    # ---- assemble the rolling training panel ----
    past = [d for d in rebdates if d < rebdate][-(n_train + 1):]
    if len(past) < 3:
        _empty()
        return None

    X_blocks, y_blocks, grp = [], [], []
    for d0, d1 in zip(past[:-1], past[1:]):
        feats = jkp_asof(bs, d0, fields, lag_days)
        if feats.empty:
            continue
        lab = fwd_return(bs, d0, d1, feats.index)
        common = feats.index.intersection(lab.index)
        if len(common) < 10:
            continue
        Xb = cs_standardize(feats.reindex(common))
        yb = lab.reindex(common)
        if is_ranker:
            # XGBRanker's NDCG objective expects non-negative integer relevance
            # grades -> map the cross-section to deciles 0..9.
            yb = (yb.rank(pct=True) * 9.999).astype(int)
        elif rank_label:
            yb = yb.rank(pct=True)
        else:
            yb = cs_zscore(yb)
        X_blocks.append(Xb.to_numpy(np.float64))
        y_blocks.append(yb.to_numpy(np.float64))
        grp.append(len(common))

    if not X_blocks:
        _empty()
        return None

    X_train = np.ascontiguousarray(np.vstack(X_blocks))
    y_train = np.concatenate(y_blocks)

    # ---- fit (with simple retrain caching) ----
    state = getattr(bs, '_ml_state', None)
    pos = rebdates.index(rebdate)
    need_fit = (state is None) or (pos - state.get('fit_pos', -10**9) >= retrain) \
        or (state.get('model_type') != model_type)

    if need_fit:
        model = _build_model(model_type, **model_kwargs)
        try:
            if is_ranker:
                model.fit(X_train, y_train, group=grp)
            else:
                model.fit(X_train, y_train)
        except Exception:
            _empty()
            return None
        bs._ml_state = {'model': model, 'fit_pos': pos, 'model_type': model_type}
    else:
        model = state['model']

    # ---- predict and standardise ----
    try:
        raw = np.asarray(
            model.predict(np.ascontiguousarray(Xnow.to_numpy(np.float64))),
            dtype=np.float64,
        )
    except Exception:
        _empty()
        return None

    pred = cs_zscore(pd.Series(raw, index=Xnow.index))
    bs.optimization_data['scores'] = pd.DataFrame(
        {field_name: pred.reindex(ids).fillna(0.0).astype(np.float64)})
    return None
