"""
Experiment 5: Extended 6-Task Sequence
=======================================
Proves forgetting resistance scales with longer task sequences.
Adds QNLI and WNLI to the existing 4-task sequence.

Sequence: RTE → MRPC → CoLA → SST-2 → QNLI → WNLI
LoRA r=8 only (since we already proved Full FT forgets), 3 seeds.

Estimated time: ~2 hours on consumer GPU
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
        logging.FileHandler('logs/extended_6task.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ========== DATA LOADING ==========

def load_and_prepare_dataset(task, tokenizer, split="train", max_samples=None):
    """Load and tokenize GLUE dataset"""
    if split == "train":
        dataset = load_dataset("glue", task, split="train")
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
    else:
        dataset = load_dataset("glue", task, split="validation")

    def tokenize(batch):
        # Handle different GLUE task formats
        if task in ["rte", "mrpc", "qnli", "wnli"]:
            # Sentence pair tasks
            key1 = "sentence1" if task in ["mrpc", "rte", "wnli"] else "question"
            key2 = "sentence2" if task in ["mrpc", "rte", "wnli"] else "sentence"
            return tokenizer(
                batch[key1], batch[key2],
                max_length=128, padding="max_length", truncation=True
            )
        elif task in ["cola", "sst2"]:
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
    """Evaluate model accuracy"""
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


# ========== SEQUENTIAL TRAINING ==========

def run_6task_sequential(seed, tasks, tokenizer):
    """Train LoRA on 6 tasks sequentially"""
    logger.info(f"\n{'='*80}")
    logger.info(f"6-Task Sequential Training: LoRA r=8, Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load ALL validation sets upfront
    val_datasets = {}
    for task in tasks:
        val_datasets[task] = load_and_prepare_dataset(task, tokenizer, split="val")
        logger.info(f"  Loaded {task} validation: {len(val_datasets[task])} samples")

    accuracy_matrix = {task: {} for task in tasks}

    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")

        train_dataset = load_and_prepare_dataset(
            current_task, tokenizer, split="train",
            max_samples=2000 if current_task in ["sst2", "qnli"] else None
        )
        logger.info(f"  Training samples: {len(train_dataset)}")

        training_args = TrainingArguments(
            output_dir=f"checkpoints/6task/{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",  # No disk bloat
            fp16=True,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        trainer.train()
        train_time = time.time() - start
        logger.info(f"  Trained in {train_time/60:.1f} min")

        # Evaluate on ALL tasks seen so far
        logger.info(f"  Evaluating on all {i+1} tasks:")
        for j, eval_task in enumerate(tasks[:i+1]):
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc

            if j < i:
                initial = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgot = initial - acc
                logger.info(f"    {eval_task}: {acc:.4f} (Initial: {initial:.4f}, Forgot: {forgot:.4f})")
            else:
                logger.info(f"    {eval_task}: {acc:.4f} (just trained)")

        del trainer
        torch.cuda.empty_cache()

    # Calculate forgetting — now for 5 prior tasks instead of 3
    forgetting_per_task = {}
    for task in tasks[:-1]:
        initial = accuracy_matrix[task][f"after_{task}"]
        final = accuracy_matrix[task][f"after_{tasks[-1]}"]
        forgetting_per_task[task] = initial - final

    metrics = {
        "avg_forgetting": float(np.mean(list(forgetting_per_task.values()))),
        "max_forgetting": float(np.max(list(forgetting_per_task.values()))),
        "forgetting_per_task": {k: round(v, 4) for k, v in forgetting_per_task.items()},
        "final_avg_accuracy": float(np.mean([accuracy_matrix[t][f"after_{tasks[-1]}"] for t in tasks])),
        "initial_avg_accuracy": float(np.mean([accuracy_matrix[t][f"after_{t}"] for t in tasks])),
        # Track forgetting trajectory: how does avg forgetting evolve as tasks increase
        "forgetting_trajectory": {},
    }

    # Compute forgetting after each task addition
    for step in range(1, len(tasks)):
        current = tasks[step]
        step_forgettings = []
        for prev_task in tasks[:step]:
            initial = accuracy_matrix[prev_task][f"after_{prev_task}"]
            after = accuracy_matrix[prev_task][f"after_{current}"]
            step_forgettings.append(initial - after)
        metrics["forgetting_trajectory"][f"after_{current}"] = {
            "avg_forgetting": round(float(np.mean(step_forgettings)), 4),
            "n_prior_tasks": step,
        }

    return accuracy_matrix, metrics


# ========== MAIN ==========

def run_extended_6task():
    logger.info("=" * 100)
    logger.info("EXPERIMENT 5: Extended 6-Task Sequence")
    logger.info("Sequence: RTE → MRPC → CoLA → SST-2 → QNLI → WNLI")
    logger.info("=" * 100)

    results_dir = Path("results/extended_6task")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    tasks = ["rte", "mrpc", "cola", "sst2", "qnli", "wnli"]
    seeds = [42, 43, 44]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    all_results = {}
    start_time = time.time()

    for seed in seeds:
        logger.info(f"\n{'#'*80}")
        logger.info(f"6-Task: Seed {seed}")
        logger.info(f"{'#'*80}")

        try:
            exp_start = time.time()
            accuracy_matrix, metrics = run_6task_sequential(seed, tasks, tokenizer)
            exp_time = (time.time() - exp_start) / 3600

            result = {
                "method": "lora_r8",
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "task_sequence": tasks,
                "n_tasks": len(tasks),
                "accuracy_matrix": accuracy_matrix,
                "forgetting_metrics": metrics,
                "time_hours": round(exp_time, 4),
            }
            all_results[f"seed{seed}"] = result

            with open(results_dir / f"6task_seed{seed}.json", 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"\n✅ Seed {seed}: Avg Forgetting={metrics['avg_forgetting']:.4f}, "
                       f"Accuracy={metrics['final_avg_accuracy']:.4f}")

            # Print trajectory
            logger.info("  Forgetting trajectory:")
            for step, vals in metrics["forgetting_trajectory"].items():
                logger.info(f"    {step}: avg_forg={vals['avg_forgetting']:.4f} "
                          f"(over {vals['n_prior_tasks']} prior tasks)")

        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 3600

    # Aggregate
    if all_results:
        forgettings = [r["forgetting_metrics"]["avg_forgetting"] for r in all_results.values()]
        accuracies = [r["forgetting_metrics"]["final_avg_accuracy"] for r in all_results.values()]

        summary = {
            "experiment": "Extended 6-Task Sequence",
            "task_sequence": tasks,
            "n_tasks": len(tasks),
            "method": "lora_r8",
            "total_time_hours": round(total_time, 2),
            "successful": len(all_results),
            "aggregate": {
                "avg_forgetting_mean": round(float(np.mean(forgettings)), 4),
                "avg_forgetting_std": round(float(np.std(forgettings)), 4),
                "final_accuracy_mean": round(float(np.mean(accuracies)), 4),
                "final_accuracy_std": round(float(np.std(accuracies)), 4),
            },
            "results": all_results,
        }

        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n\n" + "=" * 70)
        print("6-TASK SEQUENCE RESULTS (LoRA r=8)")
        print("=" * 70)
        print(f"Sequence: {' → '.join(tasks)}")
        print(f"Avg Forgetting: {np.mean(forgettings):.4f} ± {np.std(forgettings):.4f}")
        print(f"Final Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print("-" * 55)
        print("Comparison (4-task LoRA from Phase 2):")
        print(f"  4-task LoRA forgetting: ~0.6% ± 1.5%")
        print(f"  6-task LoRA forgetting: {np.mean(forgettings)*100:.1f}% ± {np.std(forgettings)*100:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    run_extended_6task()
