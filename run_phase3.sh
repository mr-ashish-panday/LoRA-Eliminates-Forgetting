#!/bin/bash
# ============================================================
# Phase 3: Generalization Experiments — Overnight Runner
# ============================================================
# Usage:
#   tmux new -s phase3
#   bash run_phase3.sh
#   Ctrl+B, D  (to detach)
#
# Order: RoBERTa (safe) → 6-Task (safe) → Vision (risky, last)
# ============================================================

set -e  # Exit on first error

echo "============================================================"
echo "PHASE 3: GENERALIZATION EXPERIMENTS"
echo "Start: $(date)"
echo "============================================================"

mkdir -p logs results

# ----- Experiment 4: RoBERTa (PRIORITY 1 — safe) -----
echo ""
echo "============================================================"
echo "EXP 4: RoBERTa-base Replication"
echo "Start: $(date)"
echo "============================================================"
python roberta_replication.py 2>&1 | tee -a phase3_overnight.log
echo "EXP 4 DONE: $(date)" | tee -a phase3_overnight.log

# ----- Experiment 5: 6-Task (PRIORITY 2 — safe) -----
echo ""
echo "============================================================"
echo "EXP 5: Extended 6-Task Sequence"
echo "Start: $(date)"
echo "============================================================"
python extended_6task.py 2>&1 | tee -a phase3_overnight.log
echo "EXP 5 DONE: $(date)" | tee -a phase3_overnight.log

# ----- Experiment 6: Vision (PRIORITY 3 — risky, last) -----
echo ""
echo "============================================================"
echo "EXP 6: Vision (ViT + CIFAR/SVHN)"
echo "Start: $(date)"
echo "============================================================"
python vision_vit_experiment.py 2>&1 | tee -a phase3_overnight.log
echo "EXP 6 DONE: $(date)" | tee -a phase3_overnight.log

echo ""
echo "============================================================"
echo "ALL PHASE 3 EXPERIMENTS COMPLETE: $(date)"
echo "============================================================"
echo "Results saved to:"
echo "  results/roberta_replication/summary.json"
echo "  results/extended_6task/summary.json"
echo "  results/vision_vit/summary.json"
