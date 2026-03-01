"""
ETH Multi-Exchange Log Price Pipeline
Dissertation: GSADF Bubble Detection in ETH (2016-06-09 to 2025-12-31)
Exchanges: Binance ETHUSDT, Kraken ETHUSD, Coinbase ETHUSD, OKX ETHUSDT
Note: CoinAPI uses exchange prefix `OKEX` for OKX spot symbols.

TIMEZONE STANDARDIZATION: All timestamps are UTC-aware for consistent
cross-exchange and cross-source alignment. This ensures proper temporal
ordering and facilitates meta-analysis across datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
START       = "2016-06-09"  # Coinbase first valid listing date in this panel
END         = "2025-12-31"

# CoinAPI config — provide key through environment variable.
COINAPI_KEY_ENV = "COINAPI_KEY"
COINAPI_BASE = "https://rest.coinapi.io/v1/ohlcv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ─────────────────────────────────────────────────────────────
# 1. FETCH DAILY CLOSES FROM COINAPI
#    Each request: period_id=1DAY, limit=10000 (~27 yrs)
# ─────────────────────────────────────────────────────────────
import requests

def fetch_coinapi(symbol_id: str, start: str, end: str, key: str) -> pd.Series:
    """
    Fetches daily OHLCV from CoinAPI for a given symbol_id.
    Returns a Series of close prices indexed by UTC date.
    symbol_id examples:
        BINANCE_SPOT_ETH_USDT
        KRAKEN_SPOT_ETH_USD
        COINBASE_SPOT_ETH_USD
        BITFLYER_SPOT_ETH_JPY
        OKEX_SPOT_ETH_USDT
    """
    url = f"{COINAPI_BASE}/{symbol_id}/history"
    params = {
        "period_id": "1DAY",
        "time_start": f"{start}T00:00:00",
        "time_end":   f"{end}T23:59:59",
        "limit": 3700,            # ~10 years of daily bars
    }
    headers = {"X-CoinAPI-Key": key}

    all_bars = []
    while True:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        bars = r.json()
        if not bars:
            break
        all_bars.extend(bars)
        # CoinAPI paginates: advance start to last bar + 1 day
        last_ts = bars[-1]["time_period_start"]
        next_start = (pd.Timestamp(last_ts) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        if next_start[:10] > end:
            break
        params["time_start"] = next_start

    if not all_bars:
        return pd.Series(dtype=float, name=symbol_id)

    df = pd.DataFrame(all_bars)
    df["date"] = pd.to_datetime(df["time_period_start"], utc=True).dt.normalize()
    df = df.set_index("date")["price_close"]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.name = symbol_id
    return df


def get_coinapi_key(env_var: str = COINAPI_KEY_ENV) -> str:
    """Fetch CoinAPI key from environment and raise a clear error if unset."""
    key = os.getenv(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {env_var} before running."
        )
    return key


# ─────────────────────────────────────────────────────────────
# 2. LOAD / ASSEMBLE RAW CLOSES
# ─────────────────────────────────────────────────────────────
def load_all_series(key: str) -> pd.DataFrame:
    """
    Fetches all exchange series.
    Returns a wide DataFrame of raw (non-log) USD closes.
    All series are UTC-aware for consistent temporal alignment.
    """
    print("Fetching Binance ETHUSDT …")
    binance = fetch_coinapi("BINANCE_SPOT_ETH_USDT", START, END, key)

    print("Fetching Kraken ETHUSD …")
    kraken  = fetch_coinapi("KRAKEN_SPOT_ETH_USD",   START, END, key)

    print("Fetching Coinbase ETHUSD …")
    coinbase = fetch_coinapi("COINBASE_SPOT_ETH_USD", START, END, key)

    print("Fetching OKX ETHUSDT …")
    okx = fetch_coinapi("OKEX_SPOT_ETH_USDT", START, END, key)

    raw = pd.DataFrame({
        "binance":  binance,
        "kraken":   kraken,
        "coinbase": coinbase,
        "okx":      okx,
    })
    return raw


# ─────────────────────────────────────────────────────────────
# 3. DATE ALIGNMENT + FORWARD-FILL
#    Strategy for early 2015 data gaps (see note below)
# ─────────────────────────────────────────────────────────────
def align_and_fill(raw: pd.DataFrame) -> pd.DataFrame:
    """
    1. Reindex to full calendar range (UTC-aware).
    2. Forward-fill up to a max of 3 consecutive missing days
       (captures weekends / exchange downtime without propagating
        genuine structural gaps in the 2015 pre-listing period).
    3. Columns that have NO data before a threshold are left as NaN
       so the composite uses only available series.
    """
    full_idx = pd.date_range(start=START, end=END, freq="D", name="date", tz="UTC")
    raw = raw.reindex(full_idx)

    # Forward-fill short gaps (≤3 days); leaves long early gaps as NaN
    raw = raw.ffill(limit=3)
    return raw


# ─────────────────────────────────────────────────────────────
# 4. LOG PRICES
# ─────────────────────────────────────────────────────────────
def compute_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Natural log of prices; drops zero/negative (shouldn't occur but safeguard)."""
    prices = prices.where(prices > 0)          # mask invalid
    logs = np.log(prices)
    logs.columns = [f"{c}_log" for c in prices.columns]
    return logs


# ─────────────────────────────────────────────────────────────
# 5. COMPOSITE LOG (row-wise mean, ignoring NaN)
# ─────────────────────────────────────────────────────────────
def add_composite(logs: pd.DataFrame) -> pd.DataFrame:
    """Append row-wise composite log price as the mean across exchanges."""
    logs["composite_log"] = logs.mean(axis=1)   # pandas mean skips NaN by default
    return logs


# ─────────────────────────────────────────────────────────────
# 6. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────
def describe_series(logs: pd.DataFrame) -> None:
    """Prints T, mean log-return, annualised vol, and pairwise correlations."""
    core_cols = [c for c in logs.columns if c != "composite_log"]
    prices_log = logs[core_cols].dropna(how="all")

    print("\n" + "═" * 60)
    print("DESCRIPTIVE STATISTICS (log price series)")
    print("═" * 60)

    log_rets = prices_log.diff().dropna(how="all")

    stats = pd.DataFrame({
        "T (obs)":        prices_log.count(),
        "First date":     prices_log.apply(lambda s: s.first_valid_index().date()),
        "Last date":      prices_log.apply(lambda s: s.last_valid_index().date()),
        "Mean log-ret":   log_rets.mean().round(6),
        "Ann. Vol (σ)":   (log_rets.std() * np.sqrt(365)).round(4),
        "Min log-price":  prices_log.min().round(4),
        "Max log-price":  prices_log.max().round(4),
    })
    print(stats.to_string())

    print("\nPairwise Pearson Correlations (log prices, overlapping obs):")
    corr = prices_log.corr().round(4)
    print(corr.to_string())
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# 7. PLOT OVERLAID LOG SERIES
# ─────────────────────────────────────────────────────────────
def plot_log_series(logs: pd.DataFrame, save_path: Path) -> None:
    """
    Overlaid log price series with composite highlighted.
    Shaded background indicates pre-Binance period (pre-2017-08).
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        "binance_log":   "#F0B90B",   # Binance gold
        "kraken_log":    "#5741D9",   # Kraken purple
        "coinbase_log":  "#0052FF",   # Coinbase blue
        "okx_log":       "#121212",   # OKX dark
    }

    for col in logs.columns:
        if col == "composite_log":
            continue
        ax.plot(logs.index, logs[col],
                label=col.replace("_log", "").capitalize(),
                color=colors.get(col, "grey"),
                linewidth=0.9, alpha=0.75)

    # Composite on top
    ax.plot(logs.index, logs["composite_log"],
            label="Composite (avg)", color="black",
            linewidth=1.8, linestyle="--", zorder=5)

    # Shade pre-Binance listing (ETH listed on Binance ~Aug 2017)
    ax.axvspan(pd.Timestamp(START, tz="UTC"), pd.Timestamp("2017-08-15", tz="UTC"),
               alpha=0.07, color="grey", label="Pre-Binance period")

    ax.set_title("ETH Log Prices (2015–2025): Multi-Exchange Panel",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("ln(ETH price, USD equivalent)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def build_gsadf_dataset(
    coinapi_key: str,
    plot_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline. Returns final DataFrame with columns:
        [binance_log, kraken_log, coinbase_log, okx_log, composite_log]
    """
    raw    = load_all_series(coinapi_key)
    filled = align_and_fill(raw)
    logs   = compute_log_prices(filled)

    logs   = add_composite(logs)
    describe_series(logs)
    plot_log_series(logs, save_path=plot_path)

    return filled, logs


# ─────────────────────────────────────────────────────────────
# SAVE + QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, log_df = build_gsadf_dataset(
        coinapi_key=get_coinapi_key(),
        plot_path=FIGURES_DIR / "eth_log_prices.png",
    )

    # Save to CSV for GSADF routine input
    out_log = DATA_RAW_DIR / "eth_log_prices.csv"
    log_df.to_csv(out_log)
    print(f"Log dataset saved → {out_log}  |  shape: {log_df.shape}")
    print(log_df.tail(5).to_string())

    out_raw = DATA_RAW_DIR / "eth_usd_utc_daily.csv"
    raw_df.to_csv(out_raw)
    print(f"Raw UTC dataset saved → {out_raw}  |  shape: {raw_df.shape}")


# ─────────────────────────────────────────────────────────────
# EARLY 2015 DATA GAP HANDLING — DISSERTATION NOTES
# ─────────────────────────────────────────────────────────────
"""
EARLY DATA GAP STRATEGY
═══════════════════════
ETH launched on mainnet 30 Jul 2015. Exchange listing timeline:
  • Kraken/Coinbase ETHUSD:  ~Aug 2015  ← earliest available
  • Binance ETHUSDT:         Aug 2017
  • OKX ETHUSDT:             Dec 2017

Recommended approach in the dissertation:
  1. FULL PANEL (2015–2025): Use composite_log as the GSADF input.
     The composite naturally handles gaps — it uses only available
     series each day (dropna mean), so early observations reflect
     Kraken/Coinbase only. Document this in §3 (Data).

  2. BALANCED PANEL (2017-08 to 2025-12): All 4 exchanges available.
     Run GSADF on this subsample as a robustness check (Table A.x).

  3. FORWARD-FILL LIMIT = 3 days: Captures exchange downtime / public
     holidays without silently propagating week-long outages during
     illiquid early periods. Days with >3 consecutive missing obs
     remain NaN and are excluded from the composite mean.

  4. REPORT per-series T in Table 1 so readers know composition changes.

  5. This script uses Binance, Kraken, Coinbase, and OKX only.
     All exchange timestamps are normalized to UTC at ingest.
"""
