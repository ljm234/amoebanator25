"""
Python orchestrator that re-emits every JSON / CSV / PNG under
outputs/. Cleaner error reporting than the bash equivalent: every step
records its name, command, exit code, duration, and a short message.

The script returns 0 only if all expected artefacts are produced. A
structured summary JSON is written to outputs/metrics/regeneration_summary.json
so CI can post a digest.

Usage:
  PYTHONPATH=. python scripts/regenerate_all_artifacts.py
  PYTHONPATH=. python scripts/regenerate_all_artifacts.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
SUMMARY_JSON = METRICS_DIR / "regeneration_summary.json"


@dataclass
class Step:
    name: str
    command: list[str]
    artifacts: list[str]
    duration_s: float = 0.0
    exit_code: int | None = None
    stdout_tail: str = ""
    artifacts_present: dict[str, bool] = field(default_factory=dict)
    artifacts_size_bytes: dict[str, int] = field(default_factory=dict)


PIPELINE: list[tuple[str, list[str], list[str]]] = [
    (
        "Train MLP + temperature + val_preds",
        [sys.executable, "-m", "ml.training_calib_dca"],
        ["outputs/model/model.pt", "outputs/model/features.json",
         "outputs/model/temperature_scale.json", "outputs/metrics/val_preds.csv",
         "outputs/metrics/metrics.json"],
    ),
    (
        "Refit Mahalanobis on train split only",
        [sys.executable, str(REPO_ROOT / "scripts" / "ood" / "refit_mahalanobis_train.py"), "--replace"],
        ["outputs/metrics/feature_stats.json",
         "outputs/metrics/feature_stats_train.json"],
    ),
    (
        "Fit logit-energy + neg-energy gates",
        [sys.executable, str(REPO_ROOT / "scripts" / "ood" / "fit_gates.py")],
        ["outputs/metrics/energy_threshold.json",
         "outputs/metrics/ood_energy.json"],
    ),
    (
        "Fit conformal qhat from probabilities",
        [sys.executable, str(REPO_ROOT / "scripts" / "conformal" / "conformal_fit_from_probs.py")],
        ["outputs/metrics/conformal.json"],
    ),
    (
        "Four-cell ablation across baselines",
        [sys.executable, str(REPO_ROOT / "scripts" / "experiments" / "run_ablation.py")],
        ["outputs/metrics/ablation_table.json",
         "outputs/metrics/ablation_table.csv"],
    ),
    (
        "Empirical coverage sweep across alpha",
        [sys.executable, str(REPO_ROOT / "scripts" / "conformal" / "eval_coverage_sweep.py")],
        ["outputs/metrics/coverage_sweep.json",
         "outputs/metrics/coverage_sweep.png"],
    ),
    (
        "ABSTAIN-rate vs accuracy Pareto",
        [sys.executable, str(REPO_ROOT / "scripts" / "conformal" / "abstain_pareto.py")],
        ["outputs/metrics/abstain_pareto.json",
         "outputs/metrics/abstain_pareto.png"],
    ),
    (
        "Synthetic OOD shift benchmarks",
        [sys.executable, str(REPO_ROOT / "scripts" / "ood" / "synthetic_ood_benchmark.py")],
        ["outputs/metrics/synthetic_ood_benchmark.json"],
    ),
]


def _check_artifacts(step: Step) -> None:
    for artifact in step.artifacts:
        p = REPO_ROOT / artifact
        present = p.exists()
        step.artifacts_present[artifact] = present
        step.artifacts_size_bytes[artifact] = int(p.stat().st_size) if present else 0


def _run_step(step: Step, dry_run: bool) -> None:
    print(f"\n=== {step.name}")
    print(f"  $ {' '.join(step.command)}")
    if dry_run:
        step.exit_code = 0
        step.stdout_tail = "(dry-run)"
        _check_artifacts(step)
        return
    started = time.time()
    proc = subprocess.run(
        step.command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    step.duration_s = round(time.time() - started, 2)
    step.exit_code = proc.returncode
    tail_lines = proc.stdout.strip().splitlines()[-5:] if proc.stdout else []
    step.stdout_tail = "\n".join(tail_lines)
    if proc.returncode != 0:
        print(f"  FAIL exit={proc.returncode}")
        print(proc.stderr[-2000:])
    else:
        print(f"  OK   exit=0 in {step.duration_s}s")
    _check_artifacts(step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip subprocess execution; only check artefact presence.")
    args = parser.parse_args(argv)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    steps: list[Step] = []
    for name, command, artifacts in PIPELINE:
        s = Step(name=name, command=command, artifacts=artifacts)
        _run_step(s, dry_run=args.dry_run)
        steps.append(s)

    # Aggregate
    n_total = len(steps)
    n_ok = sum(1 for s in steps if s.exit_code == 0)
    n_fail = n_total - n_ok
    all_artifacts = sum((s.artifacts for s in steps), [])
    n_artifacts = len(all_artifacts)
    n_present = sum(1 for s in steps for present in s.artifacts_present.values() if present)

    summary = {
        "n_steps": n_total,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_artifacts_expected": n_artifacts,
        "n_artifacts_present": n_present,
        "steps": [asdict(s) for s in steps],
    }
    def _sanitize_paths(obj: Any) -> Any:
        repo, exe = str(REPO_ROOT), sys.executable
        if isinstance(obj, str):
            return obj.replace(repo + "/", "").replace(repo, ".").replace(exe, "python")
        if isinstance(obj, dict):
            return {_sanitize_paths(k): _sanitize_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_paths(x) for x in obj]
        return obj

    summary = _sanitize_paths(summary)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))

    print()
    print("=" * 72)
    print(f"  Steps:     {n_ok}/{n_total} OK  ({n_fail} failed)")
    print(f"  Artefacts: {n_present}/{n_artifacts} present")
    print(f"  Summary written to: {SUMMARY_JSON.relative_to(REPO_ROOT)}")
    print("=" * 72)

    return 0 if (n_fail == 0 and n_present == n_artifacts) else 1


if __name__ == "__main__":
    sys.exit(main())
