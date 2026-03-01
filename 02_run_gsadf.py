"""
GSADF + BSADF Bubble Detection — Phillips, Shi & Yu (2015)
===========================================================
Reference: Phillips, P.C.B., Shi, S., & Yu, J. (2015).
           Testing for Multiple Explosive Periods in Financial Time Series.
           Journal of the American Statistical Association, 110(512), 1201–1226.

Pipeline:
  1. Compute OLS t-statistic for ADF regression on window [r1, r2]
  2. SADF  = sup over r2 ∈ [r0, 1] of ADF(0, r2)
  3. GSADF = sup over r1 < r2,  r2-r1 ≥ r0 of ADF(r1, r2)
  4. BSADF(r2) = sup over r1 ∈ [0, r2-r0] of ADF(r1, r2)   ← date-stamping
  5. Wild bootstrap (Rademacher ±1) CVs for GSADF and BSADF sequence
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from pathlib import Path
import argparse
import pickle
import time
from joblib import Parallel, delayed
import multiprocessing as mp

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# 0. MINIMAL DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    def tqdm(x, **kw):
        """Fallback iterator when tqdm is unavailable."""
        return x
    _TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. ADF REGRESSION CORE  (vectorised OLS, no statsmodels overhead)
# ─────────────────────────────────────────────────────────────────────────────
# ────────────...   
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
tbl = build_results_table(results)
    return results, tbl


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRINT + SAVE TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_results_table(tbl: pd.DataFrame) -> None:
    """Print formatted GSADF summary tables to stdout."""
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)

    print("\n" + "═" * 110)
    print("TABLE 2 — GSADF Results with Wild Bootstrap Critical Values")
    print("(Phillips–Shi–Yu 2015; regression='ct'; BIC lag selection; ")
    print("Rademacher wild bootstrap B=1499)")
    print("─" * 110)
    # Print wide columns separately for readability
    core = tbl[["Exchange", "T", "r0", "GSADF stat",
                "CV 90%", "CV 95%", "Reject H0 95%", "N Bubbles"]]
    print(core.to_string(index=False))
    print("\nBubble Periods & Aligned Events:")
    for _, row in tbl.iterrows():
        print(f"  {row['Exchange']:12s}: {row['Bubble Periods']}")
        print(f"  {'':12s}  ↳ {row['Aligned Events']}")
    print("═" * 110 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Command-line argument parsing ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="GSADF + BSADF Bubble Detection (Phillips–Shi–Yu 2015) with wild bootstrap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--B",
        type=int,
        default=1499,
        help="Number of wild bootstrap replications (use 999 for quick test run)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=12,
        help="Maximum ADF lag order for BIC selection",
    )
    parser.add_argument(
        "--min-bubble-days",
        type=int,
        default=5,
        help="Minimum consecutive exceedance days to count as bubble episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    args = parser.parse_args()

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load from CSV by default when running standalone
    df = pd.read_csv(DATA_RAW_DIR / "eth_log_prices.csv", index_col=0, parse_dates=True)

    # ── Run full analysis ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"GSADF + BSADF Analysis with Wild Bootstrap (PS&Y 2015)")
    print(f"{'═'*60}")
    print(f"Parameters: B={args.B}, max_lag={args.max_lag}, "
          f"min_bubble_days={args.min_bubble_days}, seed={args.seed}")
    print(f"{'═'*60}\n")

    results, tbl = run_gsadf_analysis(
        df,                      # log-price DataFrame
        B=args.B,                # bootstrap replications
        max_lag=args.max_lag,
        min_bubble_days=args.min_bubble_days,
        seed=args.seed,
    )

    # ── Output ────────────────────────────────────────────────────────────
    print_results_table(tbl)
    table_path = OUTPUT_TABLES_DIR / "table2_gsadf_results.csv"
    tbl.to_csv(table_path, index=False)

    # Build a common date index spanning all series for the plot
    all_dates = df.index
    fig_path = FIGURES_DIR / "fig4_bsadf.png"
    plot_bsadf(results, all_dates, save_path=str(fig_path))

    print("Done. Files written:")
    print(f"  {table_path}")
    print(f"  {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE NOTE
# ─────────────────────────────────────────────────────────────────────────────
"""
RUNTIME ESTIMATE & OPTIMISATION TIPS
══════════════════════════════════════
T ≈ 3,800 (full 2015–2025):
  • GSADF/BSADF loop:    ~5–15 min per series  (pure Python OLS, all windows)
  • Wild bootstrap B=1499: ~8–20 min per series (each rep runs full BSADF)
  • 5 series total:       ~65–175 min on a modern laptop

Speed-up options (in order of impact):
  1. numba JIT: decorate _ols_tstat and _build_adf_regressors with @njit
     → 10–30× speedup; install: pip install numba
  2. Reduce B to 999 for drafts; use 1999 for final submission run
  3. Use composite_log only for bootstrap CVs; apply same CVs to all
     exchanges (valid if series are co-integrated, i.e. ρ > 0.99)
  4. Parallel bootstrap with joblib:
       from joblib import Parallel, delayed
       gsadf_boot = Parallel(n_jobs=-1)(
           delayed(_one_bootstrap_rep)(...) for b in range(B))
  5. Trim analysis to 2017-08 onward (balanced panel) for robustness
     check — halves T and cuts runtime by ~4×

Numba stub (drop-in replacement for _ols_tstat):
  from numba import njit
  @njit(cache=True)
  def _ols_tstat_fast(Y, X):
      beta = np.linalg.lstsq(X, Y)[0]   # numba supports this
      ...  # same logic as _ols_tstat
"""