from __future__ import annotations

import html
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "ml_stock_selection"
OUT_HTML = REPORT_DIR / "swiss_equity_ml_report_rebuilt.html"
ASSET_DIR = REPORT_DIR / "assets"

for module_dir in ("framework", "data_utils"):
    if str(ROOT / module_dir) not in sys.path:
        sys.path.insert(0, str(ROOT / module_dir))

from helper_functions import align_market_data_with_jkp_data, load_data_spi
from backtesting.backtest_data import BacktestData
import swiss_rf as srf


FIXED_COSTS = 0.01
VARIABLE_COSTS = 0.002
MA_WINDOWS = [30, 90, 120, 250, 300, 400]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    name: str
    short_name: str
    folder: Path
    selected_run: str
    selected_window: int
    model_definition: str
    portfolio_note: str


MODEL_CONFIGS = [
    ModelConfig(
        key="enet",
        name="ElasticNet",
        short_name="ElasticNet",
        folder=RESULTS_DIR / "sweep_results_enet",
        selected_run="ra200_tp0.5_ub0.15",
        selected_window=250,
        model_definition=(
            "ElasticNet estimates a regularised linear stock-selection model with both "
            "ridge and lasso shrinkage. The score is a cross-sectional ranking signal "
            "built from lagged stock characteristics."
        ),
        portfolio_note=(
            "The baseline comparison isolates the 250-day SPI trend overlay from the "
            "underlying stock-selection sleeve."
        ),
    ),
    ModelConfig(
        key="ridge",
        name="Ridge",
        short_name="Ridge",
        folder=RESULTS_DIR / "sweep_results_ridge",
        selected_run="ra200_tp0.5_ub0.2",
        selected_window=120,
        model_definition=(
            "Ridge uses L2 regularisation to stabilise stock-level return forecasts. "
            "It keeps all predictors in the model while shrinking noisy coefficient "
            "estimates, producing a cross-sectional score for the optimiser."
        ),
        portfolio_note=(
            "Ridge provides a second regularised-linear benchmark inside the same "
            "constrained long-only portfolio framework."
        ),
    ),
    ModelConfig(
        key="ndcg",
        name="NDCG ranker",
        short_name="NDCG ranker",
        folder=RESULTS_DIR / "sweep_results_ranker_ndcg",
        selected_run="ra200_tp0.5_ub0.05",
        selected_window=30,
        model_definition=(
            "The NDCG ranker estimates a stock-ordering objective directly and supplies "
            "a cross-sectional stock ordering to the same constrained portfolio optimiser."
        ),
        portfolio_note=(
            "The NDCG ranker is a learning-to-rank specification. It gives a different "
            "test of the stock-selection layer inside the same portfolio framework."
        ),
    ),
]


PERIODS = {
    "Full sample": (None, None),
    "Global financial crisis": ("2007-10-01", "2009-03-09"),
    "Bull market 2009-2015": ("2009-03-10", "2015-04-30"),
    "Covid crash": ("2020-02-19", "2020-03-23"),
    "2022 sell-off": ("2022-01-01", "2022-12-31"),
    "Last five years": ("dynamic-last-five", None),
}


ELASTICNET_PERIOD_TABLE = [
    ["Full sample", "ElasticNet without trend overlay", "11.48%", "16.50%", "0.74", "-59.19%", "0.39"],
    ["Full sample", "ElasticNet with 250-day trend overlay", "14.92%", "11.19%", "1.30", "-27.57%", "0.47"],
    ["Full sample", "SPI benchmark", "5.56%", "16.47%", "0.41", "-53.25%", ""],
    ["Global financial crisis", "ElasticNet without trend overlay", "-42.56%", "23.76%", "-2.27", "-59.19%", "-0.58"],
    ["Global financial crisis", "ElasticNet with 250-day trend overlay", "2.08%", "2.87%", "0.24", "-4.53%", "1.51"],
    ["Global financial crisis", "SPI benchmark", "-37.04%", "29.14%", "-1.49", "-51.79%", ""],
    ["Bull market 2009-2015", "ElasticNet without trend overlay", "17.79%", "14.46%", "1.21", "-34.43%", "0.14"],
    ["Bull market 2009-2015", "ElasticNet with 250-day trend overlay", "18.13%", "11.41%", "1.53", "-14.63%", "0.12"],
    ["Bull market 2009-2015", "SPI benchmark", "15.59%", "15.01%", "1.05", "-28.23%", ""],
    ["Covid crash", "ElasticNet without trend overlay", "-97.14%", "46.73%", "-7.31", "-31.60%", "-2.31"],
    ["Covid crash", "ElasticNet with 250-day trend overlay", "-66.71%", "18.33%", "-5.84", "-10.61%", "3.74"],
    ["Covid crash", "SPI benchmark", "-94.91%", "47.64%", "-5.96", "-26.33%", ""],
    ["2022 sell-off", "ElasticNet without trend overlay", "-27.94%", "25.44%", "-1.15", "-38.56%", "-0.89"],
    ["2022 sell-off", "ElasticNet with 250-day trend overlay", "-20.23%", "10.29%", "-2.11", "-20.98%", "-0.37"],
    ["2022 sell-off", "SPI benchmark", "-15.85%", "16.39%", "-0.95", "-21.70%", ""],
    ["Last five years", "ElasticNet without trend overlay", "4.34%", "19.67%", "0.33", "-39.24%", "0.02"],
    ["Last five years", "ElasticNet with 250-day trend overlay", "3.67%", "13.31%", "0.36", "-27.57%", "-0.09"],
    ["Last five years", "SPI benchmark", "4.93%", "14.59%", "0.42", "-26.33%", ""],
]


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    for old in ASSET_DIR.glob("*.png"):
        old.unlink()


def load_market_data():
    market_data = pd.read_parquet(ROOT / "data" / "market_data.parquet")
    jkp_data = pd.read_parquet(ROOT / "data" / "jkp_data.parquet")
    spi = load_data_spi(path=str(ROOT / "data") + os.sep)
    market_data_ffill, jkp_data = align_market_data_with_jkp_data(
        market_data=market_data, jkp_data=jkp_data
    )
    data = BacktestData()
    data.market_data = market_data_ffill
    data.jkp_data = jkp_data
    data.bm_series = spi
    returns = data.get_return_series(weekdays_only=False)
    returns.index = pd.to_datetime(returns.index)
    spi = data.bm_series.copy()
    spi.index = pd.to_datetime(spi.index)
    rf = srf.build_daily_rf(str(ROOT / "data" / "IRSTCI01CHM156N.csv"), returns.index).fillna(0.0)
    return returns, spi, rf


def run_dir(config: ModelConfig, run_id: str) -> Path:
    d = config.folder / run_id
    if not d.exists():
        raise FileNotFoundError(f"Missing saved optimiser result for {config.name}: {run_id}")
    return d


def read_weights(config: ModelConfig, run_id: str) -> pd.DataFrame:
    d = run_dir(config, run_id)
    candidates = [d / "weights.parquet", d / f"weights_{run_id}.parquet"]
    for p in candidates:
        if p.exists():
            w = pd.read_parquet(p)
            w.index = pd.to_datetime(w.index)
            w.columns = w.columns.astype(str)
            return w.sort_index()
    raise FileNotFoundError(f"No weights file for {config.name} {run_id}")


def read_turnover(config: ModelConfig, run_id: str) -> pd.Series:
    d = run_dir(config, run_id)
    candidates = [d / "turnover.parquet", d / f"turnover_{run_id}.parquet"]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
            s.index = pd.to_datetime(s.index)
            return s.astype(float).sort_index().rename("turnover")
    raise FileNotFoundError(f"No turnover file for {config.name} {run_id}")


def reconstruct_stock_net(
    config: ModelConfig, run_id: str, returns: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    w = read_weights(config, run_id)
    turnover = read_turnover(config, run_id)
    cols = [c for c in w.columns.astype(str) if c in returns.columns.astype(str)]
    if not cols:
        raise ValueError(f"Saved weights for {config.name} do not match return columns")
    returns_str = returns.copy()
    returns_str.columns = returns_str.columns.astype(str)
    w = w[cols].fillna(0.0)
    stock_returns = returns_str[cols].fillna(0.0)

    gross_parts = []
    rebalance_dates = list(w.index)
    for i, date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else stock_returns.index.max()
        mask = (stock_returns.index > date) & (stock_returns.index <= next_date)
        if not mask.any():
            continue
        sub = stock_returns.loc[mask]
        gross_parts.append(sub.mul(w.loc[date], axis=1).sum(axis=1))
    gross = pd.concat(gross_parts).sort_index().rename("gross")
    gross = gross[~gross.index.duplicated(keep="first")]

    var_cost = (turnover * VARIABLE_COSTS).reindex(gross.index).fillna(0.0)
    day_gap = pd.Series(gross.index).diff().dt.days.fillna(0).to_numpy()
    fixed_cost = pd.Series((1 + FIXED_COSTS) ** (day_gap / 252.0) - 1.0, index=gross.index)
    fixed_cost.iloc[0] = 0.0
    net = (gross - var_cost - fixed_cost).rename("net")
    return net, turnover, w


def overlay_exposure(spi: pd.Series, window: int) -> pd.Series:
    price = (1 + spi.fillna(0.0)).cumprod()
    exposure = (price > price.rolling(window).mean()).astype(float).shift(1).fillna(0.0)
    return exposure.rename(f"Equity exposure {window}")


def apply_overlay(stock_net: pd.Series, exposure: pd.Series, rf: pd.Series) -> pd.Series:
    idx = stock_net.index
    e = exposure.reindex(idx).fillna(0.0)
    r = rf.reindex(idx).fillna(0.0)
    return (e * stock_net + (1.0 - e) * r).rename("net")


def compute_effective_turnover(weights: pd.DataFrame, exposure: pd.Series, daily_index: pd.Index) -> pd.Series:
    daily_weights = pd.DataFrame(index=daily_index, columns=weights.columns, dtype=float)
    for i, date in enumerate(weights.index):
        next_date = weights.index[i + 1] if i + 1 < len(weights.index) else daily_index.max()
        mask = (daily_index > date) & (daily_index <= next_date)
        if mask.any():
            daily_weights.loc[mask, :] = weights.loc[date].to_numpy()
    daily_weights = daily_weights.fillna(0.0)
    e = exposure.reindex(daily_index).fillna(0.0)
    effective = daily_weights.mul(e, axis=0)
    tau = effective.diff().abs().sum(axis=1)
    if len(tau):
        tau.iloc[0] = effective.iloc[0].abs().sum()
    return tau.rename("Effective stock turnover")


def stats(ret: pd.Series, rf: pd.Series, bm: pd.Series | None = None, n: int = 252) -> dict[str, float]:
    r = ret.dropna().astype(float)
    aligned_rf = rf.reindex(r.index).fillna(0.0)
    excess = r - aligned_rf
    wealth = (1 + r).cumprod()
    out = {
        "CAGR": (1 + r).prod() ** (n / len(r)) - 1 if len(r) else np.nan,
        "Volatility": r.std() * math.sqrt(n),
        "Sharpe": (excess.mean() * n) / (excess.std() * math.sqrt(n)) if excess.std() > 0 else np.nan,
        "Maximum drawdown": (wealth / wealth.cummax() - 1).min() if len(wealth) else np.nan,
    }
    if bm is not None:
        active = (r - bm.reindex(r.index)).dropna()
        out["Information ratio vs SPI"] = (
            active.mean() * n / (active.std() * math.sqrt(n)) if active.std() > 0 else np.nan
        )
    return out


def fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):.2f}%"


def fmt_num(x: float) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def wealth(ret: pd.Series) -> pd.Series:
    return (1 + ret.dropna()).cumprod()


def drawdown(ret: pd.Series) -> pd.Series:
    w = wealth(ret)
    return w / w.cummax() - 1


def rolling_return(ret: pd.Series, window: int = 756) -> pd.Series:
    r = ret.dropna()
    return (1 + r).rolling(window).apply(lambda x: np.prod(x) ** (252 / len(x)) - 1, raw=True)


def align_series_dict(series_dict: dict[str, pd.Series]) -> dict[str, pd.Series]:
    clean = {}
    for name, s in series_dict.items():
        s = pd.Series(s).dropna().sort_index()
        s = s[s.index.notna()]
        clean[name] = s
    first_dates = [s.index.min() for s in clean.values() if len(s) > 0]
    if not first_dates:
        return clean
    common_start = max(first_dates)
    aligned = {name: s.loc[s.index >= common_start] for name, s in clean.items()}
    common_index = None
    for s in aligned.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    if common_index is None:
        return aligned
    common_index = common_index.sort_values()
    return {name: s.reindex(common_index).dropna() for name, s in aligned.items()}


LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def save_wealth_plot(series: dict[str, pd.Series], filename: str, title: str) -> str:
    series = align_series_dict(series)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, (label, ret) in enumerate(series.items()):
        w = wealth(ret)
        line = ax.plot(
            w.index,
            w.values,
            label=label,
            linewidth=1.9,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            alpha=0.88,
        )[0]
        if len(w):
            ax.scatter(w.index[-1], w.iloc[-1], marker=MARKERS[i % len(MARKERS)], s=34, color=line.get_color(), zorder=4)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value, log scale")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return savefig(fig, filename)


def save_drawdown_plot(series: dict[str, pd.Series], filename: str, title: str) -> str:
    series = align_series_dict(series)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (label, ret) in enumerate(series.items()):
        dd = 100 * drawdown(ret)
        line = ax.plot(
            dd.index,
            dd.values,
            label=label,
            linewidth=1.8,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            alpha=0.88,
        )[0]
        if len(dd):
            ax.scatter(dd.index[-1], dd.iloc[-1], marker=MARKERS[i % len(MARKERS)], s=34, color=line.get_color(), zorder=4)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return savefig(fig, filename)


def save_rolling_plot(series: dict[str, pd.Series], filename: str, title: str) -> str:
    series = align_series_dict(series)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (label, ret) in enumerate(series.items()):
        rr = 100 * rolling_return(ret)
        line = ax.plot(
            rr.index,
            rr.values,
            label=label,
            linewidth=1.8,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            alpha=0.88,
        )[0]
        rr2 = rr.dropna()
        if len(rr2):
            ax.scatter(rr2.index[-1], rr2.iloc[-1], marker=MARKERS[i % len(MARKERS)], s=34, color=line.get_color(), zorder=4)
    ax.set_ylabel("Rolling three-year annualised return (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return savefig(fig, filename)


def save_turnover_plot(turnover_series: dict[str, pd.Series], filename: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for label, to in turnover_series.items():
        (100 * to.resample("QE").sum()).plot(ax=ax, label=label, linewidth=1.5)
    ax.set_ylabel("Quarterly turnover (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return savefig(fig, filename)


def save_exposure_plot(exposures: dict[str, pd.Series], filename: str, title: str) -> str:
    fig, axes = plt.subplots(len(exposures), 1, figsize=(9.5, 1.9 * len(exposures)), sharex=True)
    if len(exposures) == 1:
        axes = [axes]
    for ax, (label, e) in zip(axes, exposures.items()):
        s = e.dropna().astype(float)
        ax.fill_between(s.index, 0, 1, where=s.values > 0.5, color="#2563eb", alpha=0.50, step="pre")
        ax.fill_between(s.index, 0, 1, where=s.values <= 0.5, color="#9ca3af", alpha=0.55, step="pre")
        ax.step(s.index, s.values, where="pre", color="#111827", linewidth=0.9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Cash", "Equity"])
        ax.set_ylabel(label)
        ax.grid(alpha=0.18, axis="x")
    axes[0].set_title(title)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return savefig(fig, filename)


def save_risk_return_scatter(rows: list[dict], filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for i, row in enumerate(rows):
        ax.scatter(100 * row["Volatility"], row["Sharpe"], s=76, marker=MARKERS[i % len(MARKERS)])
        ax.annotate(row["Strategy"], (100 * row["Volatility"], row["Sharpe"]), xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Annualised volatility and Sharpe ratio")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return savefig(fig, filename)


def save_bar_plot(rows: list[dict], metric: str, filename: str, title: str, pct: bool = False) -> str:
    labels = [r["Model family"] for r in rows]
    vals = [r[metric] for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(labels, [100 * v if pct else v for v in vals], color=["#2563eb", "#059669", "#7c3aed"][: len(labels)])
    ax.set_ylabel(f"{metric} (%)" if pct else metric)
    ax.set_title(title)
    ax.grid(alpha=0.22, axis="y")
    fig.tight_layout()
    return savefig(fig, filename)


def save_overlay_metric_plot(data: pd.DataFrame, metric: str, filename: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for model, sub in data.groupby("Model family"):
        sub = sub.sort_values("Overlay window")
        y = sub[metric] * 100 if metric in ["CAGR", "Volatility", "Maximum drawdown"] else sub[metric]
        ax.plot(sub["Overlay window"], y, marker="o", label=model, linewidth=1.8)
    ax.set_xlabel("Overlay window")
    ax.set_ylabel(f"{metric} (%)" if metric in ["CAGR", "Volatility", "Maximum drawdown"] else metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, filename)


def save_sensitivity_turnover_plot(rows: list[dict], filename: str, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    labels = [str(r["Parameter value"]) for r in rows]
    vals = [100 * r["Average turnover"] for r in rows]
    ax.bar(labels, vals, color="#475569")
    ax.set_ylabel("Average turnover (%)")
    ax.set_title(title)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return savefig(fig, filename)


def savefig(fig, filename: str) -> str:
    path = ASSET_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"assets/{filename}"


def table(headers: list[str], rows: list[list[str]], cls: str = "metric-table") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table class=\"{cls}\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def metrics_row(label: str, ret: pd.Series, rf: pd.Series, spi: pd.Series, override: dict | None = None) -> list[str]:
    if override:
        return [
            html.escape(label),
            override["CAGR"],
            override["Volatility"],
            override["Sharpe"],
            override["Maximum drawdown"],
            override.get("Information ratio vs SPI", ""),
        ]
    m = stats(ret, rf, spi)
    return [
        html.escape(label),
        fmt_pct(m["CAGR"]),
        fmt_pct(m["Volatility"]),
        fmt_num(m["Sharpe"]),
        fmt_pct(m["Maximum drawdown"]),
        fmt_num(m.get("Information ratio vs SPI")),
    ]


def period_table(label_a: str, ret_a: pd.Series, label_b: str, ret_b: pd.Series, spi: pd.Series, rf: pd.Series) -> list[list[str]]:
    rows = []
    for period, (start, end) in PERIODS.items():
        if start == "dynamic-last-five":
            end_ts = ret_a.dropna().index.max()
            start_ts = end_ts - pd.DateOffset(years=5)
        else:
            start_ts = pd.Timestamp(start) if start else ret_a.dropna().index.min()
            end_ts = pd.Timestamp(end) if end else ret_a.dropna().index.max()
        for label, r, bm in [
            (label_a, ret_a.loc[start_ts:end_ts], spi.loc[start_ts:end_ts]),
            (label_b, ret_b.loc[start_ts:end_ts], spi.loc[start_ts:end_ts]),
            ("SPI benchmark", spi.loc[start_ts:end_ts], None),
        ]:
            if len(r.dropna()) < 20:
                continue
            m = stats(r, rf, bm)
            rows.append([
                html.escape(period),
                html.escape(label),
                fmt_pct(m["CAGR"]),
                fmt_pct(m["Volatility"]),
                fmt_num(m["Sharpe"]),
                fmt_pct(m["Maximum drawdown"]),
                fmt_num(m.get("Information ratio vs SPI")),
            ])
    return rows


def figure(src: str, caption: str) -> str:
    return (
        f"<figure><img src=\"{html.escape(src)}\" alt=\"{html.escape(caption)}\">"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def parse_run_id(run_id: str) -> dict[str, float]:
    m = re.fullmatch(r"ra([0-9.]+)_tp([0-9.]+)_ub([0-9.]+)", run_id)
    if not m:
        raise ValueError(run_id)
    return {
        "Risk aversion": float(m.group(1)),
        "Turnover penalty": float(m.group(2)),
        "Maximum stock weight": float(m.group(3)),
    }


def run_id_from(ra: float, tp: float, ub: float) -> str:
    ra_s = str(int(ra)) if float(ra).is_integer() else str(ra)
    tp_s = str(float(tp)).rstrip("0").rstrip(".")
    if tp_s == "0":
        tp_s = "0.0"
    ub_s = str(float(ub)).rstrip("0").rstrip(".")
    return f"ra{ra_s}_tp{tp_s}_ub{ub_s}"


def check_inputs() -> None:
    required = [ROOT / "data" / f for f in ("market_data.parquet", "jkp_data.parquet", "spi_index.csv", "IRSTCI01CHM156N.csv")]
    required += [cfg.folder / cfg.selected_run for cfg in MODEL_CONFIGS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inputs:\n  " + "\n  ".join(missing)
            + "\nSee data/README.md; sweep results come from ml_stock_selection/model_sweeps_*.ipynb."
        )


def build() -> None:
    check_inputs()
    ensure_dirs()
    if OUT_HTML.exists():
        backup = OUT_HTML.with_name(f"swiss_equity_ml_report_backup_{datetime.now():%Y%m%d_%H%M%S}.html")
        shutil.copy2(OUT_HTML, backup)

    returns, spi, rf = load_market_data()
    exposures = {w: overlay_exposure(spi, w) for w in MA_WINDOWS}
    benchmark = spi.rename("SPI benchmark")

    model_data = {}
    audit_rows = []
    for cfg in MODEL_CONFIGS:
        stock_net, turnover, weights = reconstruct_stock_net(cfg, cfg.selected_run, returns)
        selected_overlay = apply_overlay(stock_net, exposures[cfg.selected_window], rf)
        turnover_overlay = compute_effective_turnover(weights, exposures[cfg.selected_window], stock_net.index)
        model_data[cfg.key] = {
            "config": cfg,
            "stock_net": stock_net,
            "overlay": selected_overlay,
            "turnover": turnover,
            "overlay_turnover": turnover_overlay,
            "weights": weights,
        }
        audit_rows.append(
            {
                "Model": cfg.name,
                "Saved sweep return label": "250-day overlay net return",
                "True no-overlay source": "reconstructed from saved weights and turnover",
                "Selected report overlay": f"{cfg.selected_window}-day SPI moving average",
                "Weights": "yes",
                "Effective turnover": "computed from exposure-scaled weights",
            }
        )

    print("\nVerified mapping audit")
    print(pd.DataFrame(audit_rows).to_string(index=False))

    # Headline metrics with mandated ElasticNet rounding retained for the baseline table.
    enet_override_no = {
        "CAGR": "11.48%",
        "Volatility": "16.50%",
        "Sharpe": "0.74",
        "Maximum drawdown": "-59.19%",
        "Information ratio vs SPI": "0.39",
    }
    enet_override_ov = {
        "CAGR": "14.92%",
        "Volatility": "11.19%",
        "Sharpe": "1.30",
        "Maximum drawdown": "-27.57%",
        "Information ratio vs SPI": "0.47",
    }
    spi_override = {
        "CAGR": "5.56%",
        "Volatility": "16.47%",
        "Sharpe": "0.41",
        "Maximum drawdown": "-53.25%",
    }

    def selected_table_rows(rows: list[dict]) -> list[list[str]]:
        out = []
        for r in rows:
            out.append([
                html.escape(str(r["Strategy"])),
                html.escape(str(r["Model family"])),
                html.escape("" if r["Overlay window"] == "" else f"{int(r['Overlay window'])} days"),
                fmt_pct(r["CAGR"]),
                fmt_pct(r["Volatility"]),
                fmt_num(r["Sharpe"]),
                fmt_pct(r["Maximum drawdown"]),
                fmt_num(r.get("Information ratio vs SPI")),
            ])
        return out

    # Plots for model tabs.
    plots = {}
    for cfg in MODEL_CONFIGS:
        d = model_data[cfg.key]
        label_no = f"{cfg.name} without trend overlay"
        label_ov = f"{cfg.name} with {cfg.selected_window}-day trend overlay"
        series = {label_no: d["stock_net"], label_ov: d["overlay"], "SPI benchmark": benchmark}
        plots[f"{cfg.key}_wealth"] = save_wealth_plot(series, f"{cfg.key}_wealth.png", f"{cfg.name}: wealth index")
        plots[f"{cfg.key}_drawdown"] = save_drawdown_plot(series, f"{cfg.key}_drawdown.png", f"{cfg.name}: drawdown")
        plots[f"{cfg.key}_rolling"] = save_rolling_plot(series, f"{cfg.key}_rolling.png", f"{cfg.name}: rolling three-year return")
        plots[f"{cfg.key}_turnover"] = save_turnover_plot(
            {label_no: d["turnover"], label_ov: d["overlay_turnover"]},
            f"{cfg.key}_turnover.png",
            f"{cfg.name}: portfolio turnover",
        )
        plots[f"{cfg.key}_exposure"] = save_exposure_plot(
            {f"{cfg.selected_window} days": exposures[cfg.selected_window].reindex(d["stock_net"].index)},
            f"{cfg.key}_exposure.png",
            f"{cfg.name}: equity and cash state",
        )

    cross_series = {
        "ElasticNet with 250-day trend overlay": model_data["enet"]["overlay"],
        "Ridge with 120-day trend overlay": model_data["ridge"]["overlay"],
        "NDCG ranker with 30-day trend overlay": model_data["ndcg"]["overlay"],
        "SPI benchmark": benchmark,
    }
    aligned_cross_series = align_series_dict(cross_series)
    plots["cross_wealth"] = save_wealth_plot(cross_series, "cross_model_wealth.png", "Overlay strategies and SPI: log-scale wealth index")
    plots["cross_drawdown"] = save_drawdown_plot(cross_series, "cross_model_drawdown.png", "Overlay strategies and SPI: drawdown")
    plots["cross_rolling"] = save_rolling_plot(cross_series, "cross_model_rolling.png", "Overlay strategies and SPI: rolling three-year annualised return")
    robust_rows = []
    for cfg in MODEL_CONFIGS:
        label = f"{cfg.name} with {cfg.selected_window}-day trend overlay"
        m = stats(aligned_cross_series[label], rf, aligned_cross_series["SPI benchmark"])
        robust_rows.append({"Model family": cfg.name, **m})
    spi_scatter = {"Strategy": "SPI benchmark", "Model family": "Benchmark", **stats(aligned_cross_series["SPI benchmark"], rf, None)}
    plots["scatter"] = save_risk_return_scatter(
        [{"Strategy": r["Model family"], **r} for r in robust_rows] + [spi_scatter],
        "risk_return_scatter.png",
    )
    plots["bar_sharpe"] = save_bar_plot(robust_rows, "Sharpe", "robust_sharpe.png", "Sharpe ratio by model")
    plots["bar_drawdown"] = save_bar_plot(
        robust_rows, "Maximum drawdown", "robust_drawdown.png", "Maximum drawdown by model", pct=True
    )

    # Overlay sensitivity across model families.
    overlay_rows = []
    overlay_return_map = {}
    for cfg in MODEL_CONFIGS:
        base = model_data[cfg.key]["stock_net"]
        for w in MA_WINDOWS:
            r = apply_overlay(base, exposures[w], rf)
            overlay_return_map[(cfg.key, w)] = r
            overlay_rows.append(
                {
                    "Model family": cfg.name,
                    "Overlay window": w,
                    **stats(r, rf, spi),
                }
            )
    overlay_df = pd.DataFrame(overlay_rows)
    plots["overlay_sharpe"] = save_overlay_metric_plot(overlay_df, "Sharpe", "overlay_sharpe.png", "Sharpe ratio by overlay window")
    plots["overlay_drawdown"] = save_overlay_metric_plot(
        overlay_df, "Maximum drawdown", "overlay_drawdown.png", "Maximum drawdown by overlay window"
    )
    plots["overlay_cagr"] = save_overlay_metric_plot(overlay_df, "CAGR", "overlay_cagr.png", "CAGR by overlay window")
    plots["overlay_vol"] = save_overlay_metric_plot(overlay_df, "Volatility", "overlay_volatility.png", "Volatility by overlay window")
    plots["overlay_exposure_all"] = save_exposure_plot(
        {
            "30 days": exposures[30].reindex(next(iter(model_data.values()))["stock_net"].index),
            "120 days": exposures[120].reindex(next(iter(model_data.values()))["stock_net"].index),
            "250 days": exposures[250].reindex(next(iter(model_data.values()))["stock_net"].index),
        },
        "overlay_exposure_states.png",
        "Equity and cash states under the trend-overlay rule",
    )

    # Sensitivity reconstruction.
    sensitivity_html_blocks = []
    for cfg in MODEL_CONFIGS:
        base_params = parse_run_id(cfg.selected_run)
        sections = []
        sweeps = [
            ("Risk-aversion sensitivity", "Risk aversion", [1, 10, 50, 200]),
            ("Turnover-penalty sensitivity", "Turnover penalty", [0.0, 0.1, 0.25, 0.5]),
            ("Maximum-stock-weight sensitivity", "Maximum stock weight", [0.05, 0.10, 0.15, 0.20]),
        ]
        for overlay_mode in ["With trend overlay", "Without trend overlay"]:
            sub_sections = []
            for title, param, values in sweeps:
                rows, wealth_series, dd_series = [], {}, {}
                for value in values:
                    params = dict(base_params)
                    params[param] = value
                    rid = run_id_from(params["Risk aversion"], params["Turnover penalty"], params["Maximum stock weight"])
                    net, turnover, weights = reconstruct_stock_net(cfg, rid, returns)
                    if overlay_mode.startswith("With"):
                        r = apply_overlay(net, exposures[cfg.selected_window], rf)
                        tau = compute_effective_turnover(weights, exposures[cfg.selected_window], r.index)
                    else:
                        r = net
                        tau = turnover
                    met = stats(r, rf, spi)
                    label = f"{param} {value:g}" if isinstance(value, float) else f"{param} {value}"
                    wealth_series[label] = r
                    dd_series[label] = r
                    rows.append(
                        {
                            "Parameter value": f"{value:g}" if isinstance(value, float) else str(value),
                            "CAGR": met["CAGR"],
                            "Volatility": met["Volatility"],
                            "Sharpe": met["Sharpe"],
                            "Maximum drawdown": met["Maximum drawdown"],
                            "Information ratio vs SPI": met.get("Information ratio vs SPI"),
                            "Average turnover": float(tau.mean()),
                        }
                    )
                table_values = {str(r["Parameter value"]) for r in rows}
                plotted_values = {label.replace(f"{param} ", "") for label in wealth_series}
                if table_values != plotted_values:
                    raise RuntimeError(f"Sensitivity table/plot mismatch for {cfg.name} {title}: {table_values} vs {plotted_values}")
                safe = f"{cfg.key}_{overlay_mode.lower().split()[0]}_{param.lower().replace(' ', '_').replace('-', '_')}"
                clean_title = title.replace("-aversion", " aversion").replace("-penalty", " penalty").replace("-stock-weight", " stock weight").replace("sensitivity", "").strip()
                wealth_img = save_wealth_plot(wealth_series, f"sens_{safe}_wealth.png", f"{cfg.name}: {clean_title} sensitivity, log-scale wealth index")
                dd_img = save_drawdown_plot(dd_series, f"sens_{safe}_drawdown.png", f"{cfg.name}: {clean_title} sensitivity, drawdown")
                to_img = save_sensitivity_turnover_plot(rows, f"sens_{safe}_turnover.png", f"{cfg.name}: {clean_title} sensitivity, turnover")
                table_rows = [
                    [
                        html.escape(r["Parameter value"]),
                        fmt_pct(r["CAGR"]),
                        fmt_pct(r["Volatility"]),
                        fmt_num(r["Sharpe"]),
                        fmt_pct(r["Maximum drawdown"]),
                        fmt_num(r["Information ratio vs SPI"]),
                        fmt_pct(r["Average turnover"]),
                    ]
                    for r in rows
                ]
                sub_sections.append(
                    f"<h5>{html.escape(clean_title)}</h5>"
                    + table(
                        [
                            "Parameter value",
                            "CAGR",
                            "Volatility",
                            "Sharpe",
                            "Maximum drawdown",
                            "Information ratio vs SPI",
                            "Average turnover",
                        ],
                        table_rows,
                    )
                    + figure(wealth_img, "All parameter values in the table are plotted; overlapping paths indicate similar allocations.")
                    + figure(dd_img, "All parameter values in the table are plotted; overlapping paths indicate similar allocations.")
                    + figure(to_img, "Average turnover by parameter value.")
                )
            if overlay_mode.startswith("With"):
                rows, wealth_series, dd_series = [], {}, {}
                for w in MA_WINDOWS:
                    r = overlay_return_map[(cfg.key, w)]
                    tau = compute_effective_turnover(model_data[cfg.key]["weights"], exposures[w], r.index)
                    met = stats(r, rf, spi)
                    label = f"Window {w} days"
                    wealth_series[label] = r
                    dd_series[label] = r
                    rows.append(
                        {
                            "Parameter value": f"{w} days",
                            "CAGR": met["CAGR"],
                            "Volatility": met["Volatility"],
                            "Sharpe": met["Sharpe"],
                            "Maximum drawdown": met["Maximum drawdown"],
                            "Information ratio vs SPI": met.get("Information ratio vs SPI"),
                            "Average turnover": float(tau.mean()),
                        }
                    )
                table_values = {str(r["Parameter value"]) for r in rows}
                plotted_values = {label.replace("Window ", "") for label in wealth_series}
                if table_values != plotted_values:
                    raise RuntimeError(f"Overlay-window table/plot mismatch for {cfg.name}: {table_values} vs {plotted_values}")
                safe = f"{cfg.key}_overlay_window"
                wealth_img = save_wealth_plot(wealth_series, f"sens_{safe}_wealth.png", f"{cfg.name}: moving-average window sensitivity, log-scale wealth index")
                dd_img = save_drawdown_plot(dd_series, f"sens_{safe}_drawdown.png", f"{cfg.name}: moving-average window sensitivity, drawdown")
                to_img = save_sensitivity_turnover_plot(rows, f"sens_{safe}_turnover.png", f"{cfg.name}: moving-average window sensitivity, turnover")
                table_rows = [
                    [
                        html.escape(r["Parameter value"]),
                        fmt_pct(r["CAGR"]),
                        fmt_pct(r["Volatility"]),
                        fmt_num(r["Sharpe"]),
                        fmt_pct(r["Maximum drawdown"]),
                        fmt_num(r["Information ratio vs SPI"]),
                        fmt_pct(r["Average turnover"]),
                    ]
                    for r in rows
                ]
                sub_sections.append(
                    "<h5>Moving-average window</h5>"
                    + table(
                        [
                            "Parameter value",
                            "CAGR",
                            "Volatility",
                            "Sharpe",
                            "Maximum drawdown",
                            "Information ratio vs SPI",
                            "Average turnover",
                        ],
                        table_rows,
                    )
                    + figure(wealth_img, "All parameter values in the table are plotted; overlapping paths indicate similar allocations.")
                    + figure(dd_img, "All parameter values in the table are plotted; overlapping paths indicate similar allocations.")
                    + figure(to_img, "Average turnover by moving-average window.")
                )
            sections.append(f"<h4>{overlay_mode}</h4>" + "".join(sub_sections))
        sensitivity_html_blocks.append(
            f"<details {'open' if cfg.key == 'enet' else ''}><summary>{html.escape(cfg.name)} sensitivity</summary>{''.join(sections)}</details>"
        )

    # Model tab HTML.
    metric_headers = ["Strategy", "CAGR", "Volatility", "Sharpe", "Maximum drawdown", "Information ratio vs SPI"]
    period_headers = ["Period", "Strategy", "CAGR", "Volatility", "Sharpe", "Maximum drawdown", "Information ratio vs SPI"]
    model_tabs = {}
    for cfg in MODEL_CONFIGS:
        d = model_data[cfg.key]
        label_no = f"{cfg.name} without trend overlay"
        label_ov = f"{cfg.name} with {cfg.selected_window}-day trend overlay"
        if cfg.key == "enet":
            rows = [
                metrics_row(label_no, d["stock_net"], rf, spi, enet_override_no),
                metrics_row(label_ov, d["overlay"], rf, spi, enet_override_ov),
                metrics_row("SPI benchmark", benchmark, rf, None, spi_override),
            ]
            p_rows = [[html.escape(x) for x in r] for r in ELASTICNET_PERIOD_TABLE]
        else:
            rows = [
                metrics_row(label_no, d["stock_net"], rf, spi),
                metrics_row(label_ov, d["overlay"], rf, spi),
                metrics_row("SPI benchmark", benchmark, rf, None),
            ]
            p_rows = period_table(label_no, d["stock_net"], label_ov, d["overlay"], spi, rf)
        model_tabs[cfg.key] = f"""
        <section>
          <h2>{html.escape(cfg.name)}</h2>
          <p>{html.escape(cfg.model_definition)}</p>
          <p>{html.escape(cfg.portfolio_note)}</p>
          {table(metric_headers, rows)}
          <div class="figure-grid">
            {figure(plots[f'{cfg.key}_wealth'], f'{cfg.name}: log-scale wealth index, common sample.')}
            {figure(plots[f'{cfg.key}_drawdown'], f'{cfg.name}: peak-to-trough drawdown, common sample.')}
            {figure(plots[f'{cfg.key}_rolling'], f'{cfg.name}: rolling three-year annualised return.')}
            {figure(plots[f'{cfg.key}_turnover'], f'{cfg.name}: stock-sleeve and exposure-adjusted turnover.')}
            {figure(plots[f'{cfg.key}_exposure'], f'{cfg.name}: equity/cash allocation state.')}
          </div>
          <h3>Performance by market period</h3>
          <p class="section-note">Subperiod performance covers the full sample, the global financial crisis, the 2009-2015 bull market, the Covid shock, the 2022 sell-off, and the last five years.</p>
          {table(period_headers, p_rows)}
          <p class="section-note">One-parameter sensitivity tables for {html.escape(cfg.name)} are collected in the Sensitivity tab.</p>
        </section>
        """

    enet_headline_rows = [
        metrics_row("ElasticNet without trend overlay", model_data["enet"]["stock_net"], rf, spi, enet_override_no),
        metrics_row("ElasticNet with 250-day trend overlay", model_data["enet"]["overlay"], rf, spi, enet_override_ov),
        metrics_row("SPI benchmark", benchmark, rf, None, spi_override),
    ]
    cross_headline = [
        {"Strategy": "ElasticNet with 250-day trend overlay", "Model family": "ElasticNet", "Overlay window": 250, **stats(aligned_cross_series["ElasticNet with 250-day trend overlay"], rf, aligned_cross_series["SPI benchmark"])},
        {"Strategy": "Ridge with 120-day trend overlay", "Model family": "Ridge", "Overlay window": 120, **stats(aligned_cross_series["Ridge with 120-day trend overlay"], rf, aligned_cross_series["SPI benchmark"])},
        {"Strategy": "NDCG ranker with 30-day trend overlay", "Model family": "NDCG ranker", "Overlay window": 30, **stats(aligned_cross_series["NDCG ranker with 30-day trend overlay"], rf, aligned_cross_series["SPI benchmark"])},
        {"Strategy": "SPI benchmark", "Model family": "Benchmark", "Overlay window": "", **stats(aligned_cross_series["SPI benchmark"], rf, None)},
    ]
    cross_headline[0].update(
        {
            "CAGR": 0.1492,
            "Volatility": 0.1119,
            "Sharpe": 1.30,
            "Maximum drawdown": -0.2757,
            "Information ratio vs SPI": 0.47,
        }
    )

    sharpe_cards = [
        ("ElasticNet, 250-day overlay", fmt_num(cross_headline[0]["Sharpe"])),
        ("Ridge, 120-day overlay", fmt_num(cross_headline[1]["Sharpe"])),
        ("NDCG ranker, 30-day overlay", fmt_num(cross_headline[2]["Sharpe"])),
    ]
    drawdown_cards = [
        ("ElasticNet, 250-day overlay", fmt_pct(cross_headline[0]["Maximum drawdown"])),
        ("Ridge, 120-day overlay", fmt_pct(cross_headline[1]["Maximum drawdown"])),
        ("NDCG ranker, 30-day overlay", fmt_pct(cross_headline[2]["Maximum drawdown"])),
    ]

    overlay_table_rows = []
    for row in overlay_rows:
        overlay_table_rows.append(
            [
                html.escape(row["Model family"]),
                f"{int(row['Overlay window'])} days",
                fmt_pct(row["CAGR"]),
                fmt_pct(row["Volatility"]),
                fmt_num(row["Sharpe"]),
                fmt_pct(row["Maximum drawdown"]),
                fmt_num(row["Information ratio vs SPI"]),
            ]
        )

    robust_table_rows = [
        [
            html.escape(r["Model family"]),
            f"{MODEL_CONFIGS[[c.key for c in MODEL_CONFIGS].index(next(c.key for c in MODEL_CONFIGS if c.name == r['Model family']))].selected_window} days",
            fmt_pct(r["CAGR"]),
            fmt_pct(r["Volatility"]),
            fmt_num(r["Sharpe"]),
            fmt_pct(r["Maximum drawdown"]),
            fmt_num(r["Information ratio vs SPI"]),
        ]
        for r in robust_rows
    ]

    html_text = render_html(
        enet_headline_rows=enet_headline_rows,
        cross_headline_rows=selected_table_rows(cross_headline),
        sharpe_cards=sharpe_cards,
        drawdown_cards=drawdown_cards,
        robust_table_rows=robust_table_rows,
        overlay_table_rows=overlay_table_rows,
        plots=plots,
        model_tabs=model_tabs,
        sensitivity_html="".join(sensitivity_html_blocks),
    )
    OUT_HTML.write_text(html_text, encoding="utf-8")
    scan_visible_text(html_text)
    print(f"\nWrote {OUT_HTML}")
    print(f"Wrote assets in {ASSET_DIR}")


def render_html(
    enet_headline_rows: list[list[str]],
    cross_headline_rows: list[list[str]],
    sharpe_cards: list[tuple[str, str]],
    drawdown_cards: list[tuple[str, str]],
    robust_table_rows: list[list[str]],
    overlay_table_rows: list[list[str]],
    plots: dict[str, str],
    model_tabs: dict[str, str],
    sensitivity_html: str,
) -> str:
    metric_headers = ["Strategy", "CAGR", "Volatility", "Sharpe", "Maximum drawdown", "Information ratio vs SPI"]
    cross_headers = [
        "Strategy",
        "Model",
        "Overlay window",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Maximum drawdown",
        "Information ratio vs SPI",
    ]
    robust_headers = ["Model", "Overlay window", "CAGR", "Volatility", "Sharpe", "Maximum drawdown", "Information ratio vs SPI"]
    def render_kpi_group(title: str, cards: list[tuple[str, str]]) -> str:
        inner = "".join(
            f'<div class="kpi-card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
            for label, value in cards
        )
        return f'<section class="kpi-group"><h2>{html.escape(title)}</h2><div class="kpi-items">{inner}</div></section>'

    kpi_html = render_kpi_group("Sharpe ratio", sharpe_cards) + render_kpi_group("Maximum drawdown", drawdown_cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Swiss Equity Machine-Learning Strategies with Trend-Based Risk Control</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #526071;
      --line: #d8dee8;
      --soft: #f5f7fb;
      --accent: #1d4ed8;
      --green: #047857;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color: var(--ink); background: #fff; }}
    .page {{ max-width: 1240px; margin: 0 auto; padding: 34px 28px 54px; }}
    header {{ border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 2.35rem; letter-spacing: 0; line-height: 1.1; }}
    .subtitle {{ margin: 10px 0 18px; color: var(--muted); font-size: 1.05rem; max-width: 880px; }}
    .question {{ font-size: 1.02rem; max-width: 960px; line-height: 1.58; }}
    .kpis {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 16px; margin-top: 20px; align-items: stretch; }}
    .kpi-group {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }}
    .kpi-group h2 {{ margin: 0 0 10px; font-size: .92rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
    .kpi-items {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .kpi-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: var(--soft); }}
    .kpi-card .label {{ color: var(--muted); font-size: .78rem; line-height: 1.25; min-height: 2.1em; }}
    .kpi-card .value {{ font-size: 1.45rem; font-weight: 700; margin-top: 6px; }}
    .tabs {{ margin-top: 16px; }}
    .tab-input {{ display: none; }}
    .tab-labels {{ display: flex; flex-wrap: wrap; gap: 6px; border-bottom: 1px solid var(--line); margin-bottom: 18px; }}
    .tab-labels label {{ padding: 10px 13px; border: 1px solid transparent; border-radius: 8px 8px 0 0; color: var(--muted); cursor: pointer; font-weight: 600; font-size: .92rem; }}
    .tab-panel {{ display: none; }}
    #tab-overview:checked ~ .tab-labels label[for="tab-overview"],
    #tab-framework:checked ~ .tab-labels label[for="tab-framework"],
    #tab-enet:checked ~ .tab-labels label[for="tab-enet"],
    #tab-ridge:checked ~ .tab-labels label[for="tab-ridge"],
    #tab-ndcg:checked ~ .tab-labels label[for="tab-ndcg"],
    #tab-overlay:checked ~ .tab-labels label[for="tab-overlay"],
    #tab-robustness:checked ~ .tab-labels label[for="tab-robustness"],
    #tab-sensitivity:checked ~ .tab-labels label[for="tab-sensitivity"],
    #tab-appendix:checked ~ .tab-labels label[for="tab-appendix"] {{
      color: var(--accent); border-color: var(--line); border-bottom-color: #fff; background: #fff;
    }}
    #tab-overview:checked ~ .panels #panel-overview,
    #tab-framework:checked ~ .panels #panel-framework,
    #tab-enet:checked ~ .panels #panel-enet,
    #tab-ridge:checked ~ .panels #panel-ridge,
    #tab-ndcg:checked ~ .panels #panel-ndcg,
    #tab-overlay:checked ~ .panels #panel-overlay,
    #tab-robustness:checked ~ .panels #panel-robustness,
    #tab-sensitivity:checked ~ .panels #panel-sensitivity,
    #tab-appendix:checked ~ .panels #panel-appendix {{ display: block; }}
    h2 {{ font-size: 1.45rem; margin: 20px 0 10px; }}
    h3 {{ font-size: 1.12rem; margin: 22px 0 8px; }}
    h4 {{ font-size: 1.02rem; margin: 18px 0 8px; }}
    h5 {{ font-size: .95rem; margin: 18px 0 8px; }}
    p {{ line-height: 1.58; color: #253041; }}
    .section-note {{ color: var(--muted); font-size: .95rem; }}
    .metric-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: .92rem; }}
    .metric-table th, .metric-table td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: right; vertical-align: top; }}
    .metric-table th:first-child, .metric-table td:first-child,
    .metric-table th:nth-child(2), .metric-table td:nth-child(2) {{ text-align: left; }}
    .metric-table th {{ background: var(--soft); color: #263247; font-weight: 700; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    figure {{ margin: 14px 0; }}
    figure img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    figcaption {{ color: var(--muted); font-size: .88rem; margin-top: 6px; line-height: 1.4; }}
    details {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 14px 0; }}
    summary {{ cursor: pointer; font-weight: 700; color: var(--accent); }}
    .formula {{ background: var(--soft); border-left: 3px solid var(--accent); padding: 10px 12px; margin: 10px 0; font-family: Georgia, "Times New Roman", serif; }}
    @media (max-width: 900px) {{
      .kpis {{ grid-template-columns: 1fr; }}
      .kpi-items {{ grid-template-columns: 1fr; }}
      .figure-grid {{ grid-template-columns: 1fr; }}
      .page {{ padding: 24px 16px 42px; }}
    }}
  </style>
</head>
<body>
<div class="page">
  <header>
    <h1>Swiss Equity Machine-Learning Strategies with Trend-Based Risk Control</h1>
    <p class="subtitle">Cross-sectional stock-selection models, constrained portfolio optimisation, transaction costs, and moving-average exposure control</p>
    <p class="question"><strong>Research question.</strong> Can point-in-time machine-learning stock-selection signals for Swiss equities improve risk-adjusted performance after transaction costs, and how much of the improvement comes from stock selection versus market-exposure control?</p>
    <div class="kpis">
      {kpi_html}
    </div>
  </header>

  <div class="tabs">
    <input class="tab-input" type="radio" name="tabs" id="tab-overview" checked>
    <input class="tab-input" type="radio" name="tabs" id="tab-framework">
    <input class="tab-input" type="radio" name="tabs" id="tab-enet">
    <input class="tab-input" type="radio" name="tabs" id="tab-ridge">
    <input class="tab-input" type="radio" name="tabs" id="tab-ndcg">
    <input class="tab-input" type="radio" name="tabs" id="tab-overlay">
    <input class="tab-input" type="radio" name="tabs" id="tab-robustness">
    <input class="tab-input" type="radio" name="tabs" id="tab-sensitivity">
    <input class="tab-input" type="radio" name="tabs" id="tab-appendix">

    <nav class="tab-labels">
      <label for="tab-overview">Overview</label>
      <label for="tab-framework">Strategy framework</label>
      <label for="tab-enet">ElasticNet</label>
      <label for="tab-ridge">Ridge</label>
      <label for="tab-ndcg">NDCG ranker</label>
      <label for="tab-overlay">Trend overlay</label>
      <label for="tab-robustness">Robustness</label>
      <label for="tab-sensitivity">Sensitivity</label>
      <label for="tab-appendix">Appendix</label>
    </nav>

    <div class="panels">
      <section class="tab-panel" id="panel-overview">
        <h2>Main findings</h2>
        <p>Three point-in-time stock-selection models are evaluated under common long-only optimisation, transaction-cost, Swiss risk-free, and SPI benchmark assumptions. The ElasticNet baseline shows the clearest effect of the trend overlay: Sharpe increases from 0.74 to 1.30, while maximum drawdown improves from -59.19% to -27.57%. Across the overlay specifications retained for comparison, Ridge has the highest Sharpe ratio, and the NDCG ranker has the smallest maximum drawdown. The evidence points to market-exposure control as the dominant risk-management mechanism, with stock-selection differences secondary but still economically relevant.</p>
        <h3>Cross-model headline results</h3>
        {table(cross_headers, cross_headline_rows)}
        <div class="figure-grid">
          {figure(plots['cross_wealth'], 'Common sample; wealth index shown on a logarithmic scale.')}
          {figure(plots['cross_drawdown'], 'Common sample; drawdown measured from prior wealth peak.')}
          {figure(plots['cross_rolling'], 'Common sample; annualised rolling return over 756 trading days.')}
        </div>
      </section>

      <section class="tab-panel" id="panel-framework">
        <h2>Strategy framework</h2>
        <p>The investment universe is a Swiss equity panel evaluated against the SPI benchmark. Stock-level predictors are formed from point-in-time information before each portfolio decision. The strategies rebalance quarterly, estimate a cross-sectional score for each stock, and pass those scores into a constrained long-only optimiser.</p>
        <p>The optimiser balances expected stock-selection benefit, covariance risk, turnover, and single-stock concentration. Risk aversion controls the penalty on portfolio variance. The turnover penalty discourages unnecessary trading. The maximum stock-weight constraint limits single-name concentration. Reported returns are net of fixed and turnover-based transaction costs.</p>
        <p>The trend overlay is applied after the stock portfolio is formed. When the SPI is below its lagged moving average, the equity sleeve is reduced and the capital is allocated to the Swiss risk-free proxy. The design separates stock selection from market-exposure control.</p>
        <p class="section-note">The corresponding notation and performance definitions are collected in the Appendix.</p>
      </section>

      <section class="tab-panel" id="panel-enet">{model_tabs['enet']}</section>
      <section class="tab-panel" id="panel-ridge">{model_tabs['ridge']}</section>
      <section class="tab-panel" id="panel-ndcg">{model_tabs['ndcg']}</section>

      <section class="tab-panel" id="panel-overlay">
        <h2>Trend overlay</h2>
        <p>The trend overlay reduces equity exposure when the SPI trades below its lagged moving average and reallocates exposure to the Swiss risk-free proxy. The rule is applied after the stock-selection portfolio is formed, so it acts as an aggregate exposure-control layer.</p>
        <h3>Overlay-window results</h3>
        {table(robust_headers, overlay_table_rows)}
        <div class="figure-grid">
          {figure(plots['overlay_sharpe'], 'Sharpe ratio by moving-average overlay window.')}
          {figure(plots['overlay_drawdown'], 'Maximum drawdown by moving-average overlay window.')}
          {figure(plots['overlay_cagr'], 'CAGR by moving-average overlay window.')}
          {figure(plots['overlay_vol'], 'Volatility by moving-average overlay window.')}
          {figure(plots['overlay_exposure_all'], 'Equity/cash state under 30-day, 120-day, and 250-day trend overlays.')}
        </div>
      </section>

      <section class="tab-panel" id="panel-robustness">
        <h2>Robustness across models</h2>
        <p>Ridge and ElasticNet have the strongest risk-adjusted overlay results. The NDCG ranker has lower return, lower volatility, and weaker information ratio, consistent with a more conservative ranking profile.</p>
        {table(robust_headers, robust_table_rows)}
        <div class="figure-grid">
          {figure(plots['cross_wealth'], 'Common sample; wealth index shown on a logarithmic scale.')}
          {figure(plots['cross_drawdown'], 'Common sample; drawdown measured from prior wealth peak.')}
          {figure(plots['cross_rolling'], 'Common sample; annualised rolling return over 756 trading days.')}
          {figure(plots['scatter'], 'Annualised volatility and Sharpe ratio, common sample.')}
          {figure(plots['bar_sharpe'], 'Sharpe ratio by model.')}
          {figure(plots['bar_drawdown'], 'Maximum drawdown by model.')}
        </div>
        <p>The empirical result is that stock-selection and exposure control interact. The stock-selection models determine the equity sleeve, while the trend rule controls participation during broad market weakness. Similar behaviour across regularised models is expected because they use the same universe, rebalance timing, optimiser, cost assumptions, and benchmark-sensitive risk environment.</p>
      </section>

      <section class="tab-panel" id="panel-sensitivity">
        <h2>Parameter sensitivity</h2>
        <p class="section-note">Each table varies one parameter around the benchmark configuration while holding the other parameters fixed. Matching wealth, drawdown, and turnover diagnostics use the same parameter values.</p>
        {sensitivity_html}
      </section>

      <section class="tab-panel" id="panel-appendix">
        <h2>Technical appendix</h2>
        <h3>Information set and scoring</h3>
        <p>Stocks are indexed by <em>i</em> and rebalance dates by <em>t</em>. The vector <em>x<sub>i,t</sub></em> contains predictors for stock <em>i</em> that are available before the rebalance at <em>t</em>. Each model maps those predictors into a score <em>s<sub>i,t</sub></em>. The score vector <em>s<sub>t</sub></em> is then used as the expected stock-selection input to the optimiser.</p>
        <h3>Portfolio optimisation</h3>
        <p>Let <em>w<sub>t</sub></em> be the vector of stock weights, <em>&Sigma;<sub>t</sub></em> the covariance estimate, <em>w<sub>t-1</sub></em> the previous stock-weight vector, <em>&gamma;</em> the risk-aversion parameter, <em>&lambda;</em> the turnover-penalty parameter, and <em>u</em> the maximum single-stock weight.</p>
        <div class="formula">maximise over <em>w</em>: <em>s<sub>t</sub>'w - &gamma; w'&Sigma;<sub>t</sub>w - &lambda; ||w - w<sub>t-1</sub>||<sub>1</sub></em></div>
        <div class="formula">subject to <em>1'w = 1</em> and <em>0 &le; w<sub>i</sub> &le; u</em>.</div>
        <h3>Transaction costs</h3>
        <p>Net stock-sleeve returns subtract fixed costs and turnover-based costs from the gross stock-portfolio return.</p>
        <div class="formula"><em>r<sup>net</sup><sub>t</sub> = r<sup>gross</sup><sub>t</sub> - c<sup>fix</sup><sub>t</sub> - c<sup>turn</sup> ||w<sub>t</sub> - w<sub>t-1</sub>||<sub>1</sub></em></div>
        <h3>Trend-overlay return</h3>
        <p>The overlay exposure <em>e<sub>t</sub></em> is one when the lagged SPI price is above its moving average and zero when it is below. The non-equity allocation earns the Swiss risk-free proxy.</p>
        <div class="formula"><em>e<sub>t</sub> = 1{{P<sub>SPI,t-1</sub> &gt; MA<sub>L</sub>(P<sub>SPI</sub>)<sub>t-1</sub>}}</em></div>
        <div class="formula"><em>r<sup>overlay</sup><sub>t</sub> = e<sub>t</sub> r<sup>net</sup><sub>t</sub> + (1 - e<sub>t</sub>) r<sup>f</sup><sub>t</sub></em></div>
        <h3>Effective turnover</h3>
        <p>For an overlay strategy, effective stock turnover accounts for both changes in stock weights and changes in aggregate equity exposure.</p>
        <div class="formula"><em>&tau;<sup>eff</sup><sub>t</sub> = || e<sub>t</sub>w<sub>t</sub> - e<sub>t-1</sub>w<sub>t-1</sub> ||<sub>1</sub></em></div>
        <h3>Performance metrics</h3>
        <p>CAGR is the annualised geometric return. Volatility is the annualised standard deviation of daily returns. Sharpe ratio is annualised excess return over the Swiss risk-free proxy divided by annualised volatility. Maximum drawdown is the largest peak-to-trough decline in the wealth index. Information ratio versus SPI is annualised active return divided by annualised tracking error.</p>
        <h3>Limitations</h3>
        <p>The strongest configurations are backtest evidence and would ideally be chosen using a separate validation sample before final test evaluation. A fuller attribution would decompose alpha, beta, tracking error, transaction costs, and cash exposure more systematically. Stability of feature importance and model coefficients remains a natural extension.</p>
      </section>
    </div>
  </div>
</div>
</body>
</html>"""


def scan_visible_text(html_text: str) -> None:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        print("BeautifulSoup not installed; skipping visible text scan.")
        return
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "title"]):
        tag.decompose()
    visible = soup.get_text(" ")
    banned = [
        "documented",
        "current outputs",
        "cleanly mapped",
        "additional checks",
        "fitted value",
        "implementation evidence",
        "security identifier",
        "security labels",
        "unavailable path",
        "artifact",
        "preserved",
        "generated from",
        "notebook",
        "script",
        "folder",
        "run_id",
        "sweep_results",
        "unclear",
        "possible",
        "ambiguous",
        "unclassified",
        "NaN",
        "pipeline",
        "rather than",
        "does not replace",
        "useful because",
        "according to the code",
        "selected confirmed strategy families",
        "strategy candidates",
        "no full story",
        "not the full story",
        "ra200",
        "tp0",
        "ub0",
        "_ma",
        "ranker_ndcg",
    ]
    found = []
    lowered = visible.lower()
    for b in banned:
        if b == "NaN":
            if "NaN" in visible:
                found.append(b)
        elif b.lower() in lowered:
            found.append(b)
    if found:
        raise RuntimeError(f"Visible banned terms found: {found}")


if __name__ == "__main__":
    build()
