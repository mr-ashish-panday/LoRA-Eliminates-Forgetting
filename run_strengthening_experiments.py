"""
Master Runner: Execute All Strengthening Experiments
=====================================================
Run this script to execute all 3 experiments in order.
Each experiment saves its results independently, so you can 
also run them individually if one fails.

Usage:
    python run_strengthening_experiments.py              # Run all 3
    python run_strengthening_experiments.py --exp 1      # Run only Experiment 1
    python run_strengthening_experiments.py --exp 2      # Run only Experiment 2
    python run_strengthening_experiments.py --exp 3      # Run only Experiment 3

Total estimated time: 10-16 hours (GPU dependent)
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment_1():
    """Extended Seeds: Full FT & LoRA for seeds 45-49"""
    logger.info("\n" + "="*100)
    logger.info("EXPERIMENT 1: Extended Seeds (Full FT + LoRA, seeds 45-49)")
    logger.info("="*100)
    from phase7_extended_seeds_v2 import run_extended_seeds
    run_extended_seeds()


def run_experiment_2():
    """Fine-Grained Freezing Ablation"""
    logger.info("\n" + "="*100)
    logger.info("EXPERIMENT 2: Fine-Grained Freezing Ablation (Phase Transition)")
    logger.info("="*100)
    from freezing_ablation_finegrained import run_finegrained_ablation
    run_finegrained_ablation()


def run_experiment_3():
    """EWC Baseline"""
    logger.info("\n" + "="*100)
    logger.info("EXPERIMENT 3: EWC Baseline (Fair Comparison)")
    logger.info("="*100)
    from ewc_baseline import run_ewc_baseline
    run_ewc_baseline()


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    # Parse args
    exp_num = None
    if len(sys.argv) > 2 and sys.argv[1] == "--exp":
        exp_num = int(sys.argv[2])
    
    experiments = {
        1: ("Extended Seeds (Full FT + LoRA)", run_experiment_1),
        2: ("Fine-Grained Freezing Ablation", run_experiment_2),
        3: ("EWC Baseline", run_experiment_3),
    }
    
    if exp_num:
        to_run = {exp_num: experiments[exp_num]}
    else:
        to_run = experiments
    
    start = time.time()
    logger.info(f"\n{'#'*100}")
    logger.info(f"STRENGTHENING EXPERIMENTS FOR TMLR")
    logger.info(f"Running: {', '.join(name for name, _ in to_run.values())}")
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info(f"{'#'*100}\n")
    
    results = {}
    for num, (name, func) in to_run.items():
        try:
            exp_start = time.time()
            func()
            exp_time = (time.time() - exp_start) / 3600
            results[num] = f"✅ {name} ({exp_time:.1f}h)"
        except Exception as e:
            results[num] = f"❌ {name}: {e}"
            logger.error(f"Experiment {num} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    total_time = (time.time() - start) / 3600
    
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS SUMMARY")
    print("="*80)
    for num, status in results.items():
        print(f"  Experiment {num}: {status}")
    print(f"\nTotal time: {total_time:.1f}h")
    print("="*80)
    print("\nResults saved to:")
    print("  results/phase7_extended_v2/summary.json")
    print("  results/freezing_ablation_finegrained/summary.json")
    print("  results/ewc_baseline/summary.json")
