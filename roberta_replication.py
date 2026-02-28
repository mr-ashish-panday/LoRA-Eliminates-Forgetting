"""
Experiment 4: RoBERTa-base Replication
=======================================
Proves model-agnostic generalization. Same protocol as Phase 2 but on RoBERTa-base.
Full FT vs LoRA r=8, 3 seeds, 4 GLUE tasks.

Key infrastructure notes:
- Uses AutoTokenizer (NOT BertTokenizer) for RoBERTa's BPE tokenizer
- save_strategy="no" to prevent disk bloat
- Aggressive memory cleanup between experiments

Estimated time: ~4 hours on consumer GPU
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
        logging.FileHandler('logs/roberta_replication.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "roberta-base"  # 125M params, BPE tokenizer


# ========== DATA LOADING ==========

def load_and_prepare_dataset(task, tokenizer, split="train", max_samples=None):
    """Load and tokenize dataset - handles RoBERTa's BPE tokenizer"""
    if split == "train":
        dataset = load_dataset("glue", task, split="train")
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
    else:
        dataset = load_dataset("glue", task, split="validation")

    # RoBERTa uses different column names for some GLUE tasks
    def tokenize(batch):
        if task in ["rte", "mrpc"]:
            return tokenizer(
                batch["sentence1"], batch["sentence2"],
                max_length=128, padding="max_length", truncation=True
            )
        elif task == "cola":
            return tokenizer(
                batch["sentence"],
                max_length=128, padding="max_length", truncation=True
            )
        elif task == "sst2":
            return tokenizer(
                batch["sentence"],
                max_length=128, padding="max_length", truncation=True
            )

    dataset = dataset.rename_column("label", "labels")
    cols_to_remove = [col for col in dataset.column_names if col != "labels"]
    dataset = dataset.map(tokenize, batched=True, remove_columns=cols_to_remove)
    dataset.set_format("torch")

    return dataset


# ========== EVALUATION ==========

def evaluate_on_task(model, dataset, device):
    """Evaluate model accuracy on a task"""
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
    """Create RoBERTa model with or without LoRA"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if method == "full":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        return model, "full_finetuning"

    elif method == "lora":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        # RoBERTa attention modules are named "query" and "value" same as BERT
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  LoRA: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")
        return model, "lora_r8"


# ========== SEQUENTIAL TRAINING ==========

def run_sequential_training(method, seed, tasks, tokenizer):
    """Sequential training with forgetting measurement"""
    logger.info(f"\n{'='*80}")
    logger.info(f"RoBERTa Sequential Training: Method={method}, Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")

    model, method_name = create_model(method, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load all validation sets (FULL)
    val_datasets = {}
    for task in tasks:
        val_datasets[task] = load_and_prepare_dataset(task, tokenizer, split="val")
        logger.info(f"  Loaded {task} validation: {len(val_datasets[task])} samples")

    accuracy_matrix = {task: {} for task in tasks}

    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")

        train_dataset = load_and_prepare_dataset(
            current_task, tokenizer, split="train",
            max_samples=2000 if current_task == "sst2" else None
        )
        logger.info(f"  Training samples: {len(train_dataset)}")

        training_args = TrainingArguments(
            output_dir=f"checkpoints/roberta/{method}_{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",  # CRITICAL: no disk bloat
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

        torch.cuda.reset_peak_memory_stats()
        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        logger.info(f"  Training completed in {train_time/60:.1f} min, Peak memory: {peak_mem:.2f}GB")

        logger.info(f"  Evaluating on all tasks seen so far:")
        for j, eval_task in enumerate(tasks[:i+1]):
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            if j < i:
                initial_acc = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgot = initial_acc - acc
                logger.info(f"    {eval_task}: {acc:.4f} (Initial: {initial_acc:.4f}, Forgot: {forgot:.4f})")
            else:
                logger.info(f"    {eval_task}: {acc:.4f} (just trained)")

        del trainer
        torch.cuda.empty_cache()

    forgetting_metrics = calculate_forgetting(accuracy_matrix, tasks)
    return accuracy_matrix, forgetting_metrics


def calculate_forgetting(accuracy_matrix, tasks):
    """Calculate forgetting metrics"""
    backward_transfer = []
    for i, task in enumerate(tasks[:-1]):
        initial = accuracy_matrix[task][f"after_{task}"]
        final = accuracy_matrix[task][f"after_{tasks[-1]}"]
        backward_transfer.append(initial - final)

    return {
        "avg_forgetting": float(np.mean(backward_transfer)) if backward_transfer else 0.0,
        "max_forgetting": float(np.max(backward_transfer)) if backward_transfer else 0.0,
        "forgetting_per_task": {tasks[i]: float(backward_transfer[i]) for i in range(len(backward_transfer))},
        "final_avg_accuracy": float(np.mean([accuracy_matrix[t][f"after_{tasks[-1]}"] for t in tasks])),
        "initial_avg_accuracy": float(np.mean([accuracy_matrix[t][f"after_{t}"] for t in tasks])),
    }


# ========== MAIN ==========

def run_roberta_replication():
    logger.info("=" * 100)
    logger.info("EXPERIMENT 4: RoBERTa-base Replication")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("=" * 100)

    results_dir = Path("results/roberta_replication")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    tasks = ["rte", "mrpc", "cola", "sst2"]
    methods = ["full", "lora"]
    seeds = [42, 43, 44]

    # CRITICAL: Use AutoTokenizer for RoBERTa's BPE tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_results = {}
    start_time = time.time()

    for method in methods:
        for seed in seeds:
            exp_key = f"{method}_seed{seed}"
            logger.info(f"\n{'#'*80}")
            logger.info(f"RoBERTa: {exp_key}")
            logger.info(f"{'#'*80}")

            try:
                exp_start = time.time()
                accuracy_matrix, forgetting = run_sequential_training(
                    method, seed, tasks, tokenizer
                )
                exp_time = (time.time() - exp_start) / 3600

                result = {
                    "model": MODEL_NAME,
                    "method": method,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "task_sequence": tasks,
                    "accuracy_matrix": accuracy_matrix,
                    "forgetting_metrics": {k: round(v, 4) if isinstance(v, float) else
                                           {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v
                                           for k, v in forgetting.items()},
                    "time_hours": round(exp_time, 4),
                }
                all_results[exp_key] = result

                with open(results_dir / f"roberta_{exp_key}.json", 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"\n✅ {exp_key}: Forgetting={forgetting['avg_forgetting']:.4f}, "
                          f"Accuracy={forgetting['final_avg_accuracy']:.4f}, Time={exp_time:.2f}h")

            except Exception as e:
                logger.error(f"❌ {exp_key} failed: {e}")
                logger.error(traceback.format_exc())

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 3600

    # Compute method-level summaries
    method_summaries = {}
    for method in methods:
        method_results = [v for k, v in all_results.items() if k.startswith(method)]
        if method_results:
            forgettings = [r["forgetting_metrics"]["avg_forgetting"] for r in method_results]
            accuracies = [r["forgetting_metrics"]["final_avg_accuracy"] for r in method_results]
            method_summaries[method] = {
                "avg_forgetting_mean": round(float(np.mean(forgettings)), 4),
                "avg_forgetting_std": round(float(np.std(forgettings)), 4),
                "final_accuracy_mean": round(float(np.mean(accuracies)), 4),
                "final_accuracy_std": round(float(np.std(accuracies)), 4),
                "n_seeds": len(method_results),
            }

    summary = {
        "experiment": "RoBERTa-base Replication",
        "model": MODEL_NAME,
        "total_time_hours": round(total_time, 2),
        "total_experiments": len(methods) * len(seeds),
        "successful": len(all_results),
        "method_summary": method_summaries,
        "results": all_results,
    }

    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n\n" + "=" * 70)
    print("ROBERTA REPLICATION RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} {'Forgetting':>12} {'Std':>8} {'Accuracy':>10}")
    print("-" * 55)
    for method, stats in method_summaries.items():
        print(f"{method:<25} {stats['avg_forgetting_mean']:>12.4f} "
              f"{stats['avg_forgetting_std']:>8.4f} {stats['final_accuracy_mean']:>10.4f}")
    print("-" * 55)
    print("Comparison (BERT-base from Phase 2):")
    print(f"{'BERT Full FT':<25} {'0.1986':>12} {'0.053':>8} {'0.645':>10}")
    print(f"{'BERT LoRA':<25} {'0.0056':>12} {'0.015':>8} {'0.595':>10}")
    print("=" * 70)


if __name__ == "__main__":
    run_roberta_replication()
