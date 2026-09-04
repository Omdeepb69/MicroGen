"""Master script to execute the complete empirical experiment suite, generating results/raw/experiments.jsonl, LaTeX paper tables, and PDF figures."""

import argparse
import os
import subprocess
import sys


EXPERIMENT_SCRIPTS = [
    "experiments/context_sweep.py",
    "experiments/prefix_sharing.py",
    "experiments/quant_lifecycle.py",
    "experiments/batching_concurrency.py",
    "experiments/speculative_sweep.py",
    "experiments/hardware_duality.py",
    "experiments/model_generalization.py",
    "experiments/paged_memory_pressure.py",
    "experiments/combined_interactions.py",
]


def run_suite(quick: bool = False) -> None:
    """Execute all experiment scripts in sequence to populate results/raw/experiments.jsonl."""
    print("================================================================================")
    print("      MICROGEN MASTER EMPIRICAL EXPERIMENT SUITE EXECUTION")
    print("================================================================================")
    
    # Ensure fresh output directory structure
    os.makedirs("results/raw", exist_ok=True)
    raw_jsonl = "results/raw/experiments.jsonl"
    if os.path.exists(raw_jsonl):
        os.remove(raw_jsonl)
        print(f"[+] Cleared prior raw logs: {raw_jsonl}")

    flag = ["--quick"] if quick else []

    # Configure PYTHONPATH to include current root directory and suppress progress bar pipe deadlocks
    env = os.environ.copy()
    cwd = os.getcwd()
    existing_ppath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cwd}:{existing_ppath}" if existing_ppath else cwd
    env["TQDM_DISABLE"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    for script in EXPERIMENT_SCRIPTS:
        if not os.path.exists(script):
            print(f"[!] Warning: Script missing: {script}", file=sys.stderr)
            continue

        print(f"\n---> Executing experiment module: {script} {' '.join(flag)}")
        cmd = [sys.executable, script] + flag
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"[!] Error running {script} (exit code: {res.returncode})", file=sys.stderr)
        else:
            print(f"[✓] Successfully finished: {script}")

    print("\n================================================================================")
    print("      EXPORTING PUBLICATION TABLES & VECTOR FIGURES")
    print("================================================================================")
    
    # Run exporter scripts
    subprocess.run([sys.executable, "scripts/export_paper_tables.py"], check=True, env=env)
    subprocess.run([sys.executable, "scripts/generate_paper_figures.py"], check=True, env=env)
    
    print("\n[🎉] Complete empirical experiment suite execution successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroGen Master Experiment Runner")
    parser.add_argument("--quick", action="store_true", help="Run shortened trial count for fast verification")
    args = parser.parse_args()

    run_suite(quick=args.quick)
