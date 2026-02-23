"""
Phase 7 Extended Seeds V2: Complete 8-seed validation for Full FT and LoRA
==========================================================================
The original phase7 used seeds 42-44. The extended run (phase7_extended) 
attempted seeds 45-49 but failed for Full FT and LoRA (only 4-bit succeeded).

This script re-runs ONLY Full FT and LoRA for seeds 45-49 to bring them 
to 8-seed parity with 4-bit, fixing the statistical integrity issue.

Task sequence: RTE → MRPC → CoLA → SST2
Methods: Full fine-tuning, LoRA r=8  
5 seeds × 2 methods = 10 experiments

Estimated time: 3-5 hours on consumer GPU
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time
import gc
import traceback

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/phase7_extended_v2.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ========== DATA LOADING ==========

def load_and_prepare_dataset(task, tokenizer, split="train", max_samples=None):
    """Load and tokenize dataset"""
    if split == "train":
        dataset = load_dataset("glue", task, split="train")
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
    else:
        # Use FULL validation set (not 200 subset) for better accuracy estimates
        dataset = load_dataset("glue", task, split="validation")
    
    def tokenize(batch):
        if "sentence2" in batch:
            texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
        else:
            texts = batch["sentence"]
        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
    
    dataset = dataset.rename_column("label", "labels")
    cols_to_remove = [col for col in dataset.column_names if col != "labels"]
    dataset = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
    dataset.set_format("torch")
    
    return dataset


# ========== EVALUATION ==========

def evaluate_on_task(model, dataset, device):
    """Evaluate model on a task dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**sample)
            pred = outputs.logits.argmax(dim=1).item()
            if pred == sample["labels"].item():
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


# ========== MODEL CREATION ==========

def create_model(method, seed):
    """Create model based on method type"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if method == "full":
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        return model, "full_finetuning"
    
    elif method == "lora":
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        return model, "lora_r8"


# ========== SEQUENTIAL TRAINING ==========

def run_sequential_training(method, seed, tasks, tokenizer):
    """
    Train on tasks sequentially and measure forgetting.
    Uses full validation sets for more reliable accuracy estimates.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Sequential Training: Method={method}, Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")
    
    # Create model
    model, method_name = create_model(method, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load all validation sets (FULL, not 200 subset)
    val_datasets = {}
    for task in tasks:
        val_datasets[task] = load_and_prepare_dataset(task, tokenizer, split="val")
        logger.info(f"  Loaded {task} validation: {len(val_datasets[task])} samples")
    
    # Accuracy matrix: accuracy[task_i][after_task_j]
    accuracy_matrix = {task: {} for task in tasks}
    
    # Train each task sequentially
    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")
        
        # Load training data
        train_dataset = load_and_prepare_dataset(
            current_task, 
            tokenizer, 
            split="train",
            max_samples=2000 if current_task == "sst2" else None
        )
        logger.info(f"  Training samples: {len(train_dataset)}")
        
        # Training arguments - identical to original phase7
        training_args = TrainingArguments(
            output_dir=f"checkpoints/phase7_ext_v2/{method}_{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            fp16=True,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train
        torch.cuda.reset_peak_memory_stats()
        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        logger.info(f"  Training completed in {train_time/60:.1f} min, Peak memory: {peak_memory_gb:.2f}GB")
        
        # Evaluate on ALL previous tasks (including current)
        logger.info(f"  Evaluating on all tasks seen so far:")
        for j, eval_task in enumerate(tasks[:i+1]):
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            
            if j < i:  # Previous task
                initial_acc = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgetting = initial_acc - acc
                logger.info(f"    {eval_task}: {acc:.4f} (Initial: {initial_acc:.4f}, Forgot: {forgetting:.4f})")
            else:
                logger.info(f"    {eval_task}: {acc:.4f} (just trained)")
        
        del trainer
        torch.cuda.empty_cache()
    
    # Calculate forgetting metrics
    forgetting_metrics = calculate_forgetting_metrics(accuracy_matrix, tasks)
    
    return accuracy_matrix, forgetting_metrics


def calculate_forgetting_metrics(accuracy_matrix, tasks):
    """Calculate comprehensive forgetting metrics"""
    metrics = {}
    
    # Backward Transfer (forgetting)
    backward_transfer = []
    for i, task in enumerate(tasks[:-1]):
        initial_acc = accuracy_matrix[task][f"after_{task}"]
        final_acc = accuracy_matrix[task][f"after_{tasks[-1]}"]
        forgetting = initial_acc - final_acc
        backward_transfer.append(forgetting)
    
    metrics["avg_forgetting"] = float(np.mean(backward_transfer)) if backward_transfer else 0.0
    metrics["max_forgetting"] = float(np.max(backward_transfer)) if backward_transfer else 0.0
    metrics["forgetting_per_task"] = {tasks[i]: float(backward_transfer[i]) for i in range(len(backward_transfer))}
    
    # Final average accuracy
    final_accuracies = [accuracy_matrix[task][f"after_{tasks[-1]}"] for task in tasks]
    metrics["final_avg_accuracy"] = float(np.mean(final_accuracies))
    
    # Initial average accuracy
    initial_accuracies = [accuracy_matrix[task][f"after_{task}"] for task in tasks]
    metrics["initial_avg_accuracy"] = float(np.mean(initial_accuracies))
    
    return metrics


# ========== MAIN ==========

def run_extended_seeds():
    logger.info("=" * 100)
    logger.info("PHASE 7 EXTENDED V2: Full FT and LoRA for seeds 45-49")
    logger.info("Using FULL validation sets for better accuracy estimates")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    
    results_dir = Path("results/phase7_extended_v2")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    methods = ["full", "lora"]  # Only these two need extended seeds
    seeds = [45, 46, 47, 48, 49]  # The missing seeds
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    all_results = {}
    completed = 0
    failed = 0
    total = len(methods) * len(seeds)
    start_time_overall = time.time()
    
    for method in methods:
        for seed in seeds:
            completed += 1
            logger.info(f"\n{'#'*100}")
            logger.info(f"EXPERIMENT {completed}/{total}: Method={method}, Seed={seed}")
            logger.info(f"{'#'*100}")
            
            try:
                experiment_start = time.time()
                
                accuracy_matrix, forgetting_metrics = run_sequential_training(
                    method, seed, tasks, tokenizer
                )
                
                experiment_time = (time.time() - experiment_start) / 3600
                
                result = {
                    "experiment_id": f"phase7_ext_v2_{method}_seed{seed}",
                    "phase": "7_extended_v2",
                    "method": method,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "task_sequence": tasks,
                    "accuracy_matrix": accuracy_matrix,
                    "forgetting_metrics": {
                        "avg_forgetting": round(forgetting_metrics["avg_forgetting"], 4),
                        "max_forgetting": round(forgetting_metrics["max_forgetting"], 4),
                        "forgetting_per_task": {k: round(v, 4) for k, v in forgetting_metrics["forgetting_per_task"].items()},
                        "final_avg_accuracy": round(forgetting_metrics["final_avg_accuracy"], 4),
                        "initial_avg_accuracy": round(forgetting_metrics["initial_avg_accuracy"], 4),
                    },
                    "metrics": {
                        "total_time_hours": round(experiment_time, 4),
                    }
                }
                
                all_results[f"{method}_seed{seed}"] = result
                
                # Save individual result
                result_file = results_dir / f"phase7_{method}_seed{seed}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"\n✅ EXPERIMENT COMPLETED!")
                logger.info(f"   Avg Forgetting: {forgetting_metrics['avg_forgetting']:.4f}")
                logger.info(f"   Final Avg Acc: {forgetting_metrics['final_avg_accuracy']:.4f}")
                logger.info(f"   Time: {experiment_time:.2f}h")
                
            except Exception as e:
                failed += 1
                logger.error(f"❌ Error in experiment {method}_seed{seed}: {e}")
                logger.error(traceback.format_exc())
                
                # Save error info
                error_file = results_dir / f"error_{method}_seed{seed}.json"
                with open(error_file, 'w') as f:
                    json.dump({
                        "method": method, "seed": seed,
                        "error": str(e), "traceback": traceback.format_exc()
                    }, f, indent=2)
            
            finally:
                # Aggressive cleanup between experiments
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    total_time = (time.time() - start_time_overall) / 3600
    
    # Save summary
    summary = {
        "phase": "7_extended_v2",
        "description": "Extended seeds (45-49) for Full FT and LoRA",
        "total_experiments": total,
        "successful_experiments": len(all_results),
        "failed_experiments": failed,
        "total_time_hours": round(total_time, 2),
        "methods_tested": methods,
        "seeds_tested": seeds,
        "task_sequence": tasks,
        "validation_set": "FULL (not 200 subset)",
        "results": all_results,
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 7 EXTENDED V2 COMPLETED!")
    logger.info(f"Total Time: {total_time:.2f}h")
    logger.info(f"Successful: {len(all_results)}/{total} (Failed: {failed})")
    logger.info(f"{'='*100}")
    
    # Print quick summary
    if all_results:
        print("\n\n=== QUICK SUMMARY ===")
        for key, res in all_results.items():
            fm = res["forgetting_metrics"]
            print(f"  {key}: Forgetting={fm['avg_forgetting']:.4f}, Accuracy={fm['final_avg_accuracy']:.4f}")


if __name__ == "__main__":
    run_extended_seeds()
