#!/usr/bin/env bash
# Amoebanator V1.0 - end-to-end pipeline runner.
#
# Trains the MLP, fits temperature scaling, fits both energy gates, fits
# Mahalanobis on the train split only, refits conformal qhat (with the small-
# sample warning), and runs the four-cell ablation + coverage sweep + Pareto
# figure. Verifies that every expected artefact lands on disk.
#
# Usage:
#   PYTHONPATH=. bash scripts/run_full_pipeline.sh
#
# Exit code 0 means every artefact was produced. Any non-zero exit is a hard
# failure - `set -euo pipefail` aborts on the first error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"

# --- Helpers --------------------------------------------------------------
step_count=0
step() {
    step_count=$((step_count + 1))
    local name="$1"
    local cmd="$2"
    local started_at
    started_at=$(date +%s)
    echo
    echo "====================================================================="
    echo "[STEP ${step_count}] ${name}"
    echo "  \$ ${cmd}"
    echo "====================================================================="
    eval "$cmd"
    local elapsed=$(( $(date +%s) - started_at ))
    echo "[STEP ${step_count}] DONE in ${elapsed}s"
}

# --- Pipeline -------------------------------------------------------------
step "Train MLP + temperature scaling + emit val_preds.csv" \
    "python -m ml.training_calib_dca"

step "Fit train-only Mahalanobis stats (closes audit-flagged contamination)" \
    "python scripts/ood/refit_mahalanobis_train.py --replace"

step "Fit both energy gates from real validation logits" \
    "python scripts/ood/fit_gates.py"

step "Fit conformal qhat from probabilities (held-out framework still warns at n<100)" \
    "python scripts/conformal/conformal_fit_from_probs.py"

step "Run four-cell ablation across baselines + Amoebanator MLP" \
    "python scripts/experiments/run_ablation.py"

step "Empirical coverage sweep across alpha in {0.05, 0.10, 0.20}" \
    "python scripts/conformal/eval_coverage_sweep.py"

step "Compute ABSTAIN-rate vs accuracy Pareto frontier" \
    "python scripts/conformal/abstain_pareto.py"

step "Synthetic OOD shift benchmarks (covariate + label shift)" \
    "python scripts/ood/synthetic_ood_benchmark.py"

# --- Artefact verification ------------------------------------------------
echo
echo "====================================================================="
echo "[VERIFY] expected artefacts"
echo "====================================================================="

EXPECTED=(
    "outputs/model/model.pt"
    "outputs/model/features.json"
    "outputs/model/temperature_scale.json"
    "outputs/metrics/val_preds.csv"
    "outputs/metrics/metrics.json"
    "outputs/metrics/feature_stats.json"
    "outputs/metrics/feature_stats_train.json"
    "outputs/metrics/energy_threshold.json"
    "outputs/metrics/ood_energy.json"
    "outputs/metrics/conformal.json"
    "outputs/metrics/ablation_table.json"
    "outputs/metrics/coverage_sweep.json"
    "outputs/metrics/abstain_pareto.json"
    "outputs/metrics/synthetic_ood_benchmark.json"
)

missing=0
for f in "${EXPECTED[@]}"; do
    if [[ -f "$f" ]]; then
        sz=$(stat -f "%z" "$f" 2>/dev/null || stat -c "%s" "$f")
        printf "  OK    %6d bytes  %s\n" "$sz" "$f"
    else
        printf "  MISSING            %s\n" "$f"
        missing=$((missing + 1))
    fi
done

echo
if [[ "$missing" -eq 0 ]]; then
    echo "[VERIFY] all ${#EXPECTED[@]} expected artefacts present."
    exit 0
fi
echo "[VERIFY] FAIL - ${missing} of ${#EXPECTED[@]} artefacts missing."
exit 1
