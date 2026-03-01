"""Master pipeline runner for ETH GSADF dissertation analysis."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"

STEPS = [
    {
        "name": "01_fetch_data_UTC.py",
        "script": CODE_DIR / "01_fetch_data_UTC.py",
        "expected": [
            DATA_RAW_DIR / "eth_log_prices.csv",
            DATA_RAW_DIR / "eth_usd_utc_daily.csv",
            FIGURES_DIR / "eth_log_prices.png",
        ],
    },
    {
        "name": "02_run_gsadf.py",
        "script": CODE_DIR / "02_run_gsadf.py",
        "expected": [
            OUTPUT_TABLES_DIR / "table2_gsadf_results.csv",
            FIGURES_DIR / "fig4_bsadf.png",
        ],
    },
    {
        "name": "03_cross_exchange.py",
        "script": CODE_DIR / "03_cross_exchange.py",
        "expected": [
            OUTPUT_TABLES_DIR / "table3_alignment.csv",
            OUTPUT_TABLES_DIR / "table4_discordance.csv",
            OUTPUT_TABLES_DIR / "disparity_rolling.csv",
            FIGURES_DIR / "fig5_bsadf_overlaid.png",
            FIGURES_DIR / "fig6_disparity.png",
        ],
    },
    {
        "name": "04_preliminary_analysis.py",
        "script": CODE_DIR / "04_preliminary_analysis.py",
        "expected": [
            OUTPUT_TABLES_DIR / "table1_adf_descriptive.csv",
            FIGURES_DIR / "fig1_log_prices.png",
            FIGURES_DIR / "fig2_volatility.png",
            FIGURES_DIR / "fig3_bubble_windows.png",
            FIGURES_DIR / "fig4_adf_heatmap.png",
        ],
    },
]


def run_step(script: Path, env: dict[str, str]) -> tuple[int, str, str, float]:
    """Run one script and return exit code, stdout, stderr, and runtime seconds."""
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def main() -> int:
    """Execute all pipeline steps in order with output validation and logging."""
    OUTPUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = OUTPUT_LOGS_DIR / f"run_all_{run_ts}.log"

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    overall_start = time.time()
    lines: list[str] = []
    lines.append(f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"Working dir: {PROJECT_ROOT}")

    for step in STEPS:
        step_name = step["name"]
        script = step["script"]
        expected = step["expected"]

        lines.append("\n" + "=" * 72)
        lines.append(f"STEP: {step_name}")
        lines.append("=" * 72)

        code, out, err, elapsed = run_step(script, env)
        lines.append(f"Exit code: {code}")
        lines.append(f"Runtime sec: {elapsed:.2f}")
        if out.strip():
            lines.append("-- STDOUT --")
            lines.append(out.strip())
        if err.strip():
            lines.append("-- STDERR --")
            lines.append(err.strip())

        if code != 0:
            lines.append(f"FAILED: {step_name}")
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Pipeline failed at {step_name}. See log: {log_path}")
            return code

        missing = [str(p) for p in expected if not p.exists()]
        if missing:
            lines.append("Missing expected outputs:")
            lines.extend(missing)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Pipeline failed output validation at {step_name}. See log: {log_path}")
            return 2

        lines.append("Output validation: OK")

    total_elapsed = time.time() - overall_start
    lines.append("\n" + "=" * 72)
    lines.append("PIPELINE STATUS: SUCCESS")
    lines.append(f"Total runtime sec: {total_elapsed:.2f}")
    lines.append(f"Run finished (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 72)

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Pipeline completed successfully.")
    print(f"Log: {log_path}")
    print(f"Data: {DATA_RAW_DIR}")
    print(f"Tables: {OUTPUT_TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
