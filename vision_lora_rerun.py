"""
Vision LoRA-Only Rerun (Fixed)
==============================
Full FT already completed (3 seeds). This script only runs the 3 LoRA seeds
with the TaskType bug fixed.

Bug: TaskType.SEQ_CLS wraps ViT with a text-classification forward() that 
expects 'input_ids'. Fix: omit task_type entirely so PEFT auto-detects.
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
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/vision_lora_rerun.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "google/vit-base-patch16-224-in21k"


def load_vision_dataset(task_name, processor, split="train", max_samples=None):
    """Load and preprocess vision datasets"""
    if task_name == "cifar10":
        dataset = load_dataset("cifar10", split=split)
        num_labels = 10
        image_key = "img"
        label_key = "label"
    elif task_name == "svhn":
        ds_split = "train" if split == "train" else "test"
        dataset = load_dataset("svhn", "cropped_digits", split=ds_split)
        num_labels = 10
        image_key = "image"
        label_key = "label"
    elif task_name == "cifar100":
        dataset = load_dataset("cifar100", split=split)
        num_labels = 20
        image_key = "img"
        label_key = "coarse_label"

    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)

    # Rename label column if needed
    if label_key != "labels":
        dataset = dataset.rename_column(label_key, "labels")

    def transform(batch):
        images = batch[image_key]
        processed = []
        for img in images:
            if hasattr(img, 'convert'):
                img = img.convert("RGB")
            processed.append(img)
        result = processor(images=processed, return_tensors="pt")
        return {"pixel_values": result["pixel_values"]}

    cols_to_remove = [c for c in dataset.column_names if c not in ["labels"]]
    dataset = dataset.map(transform, batched=True, batch_size=32, remove_columns=cols_to_remove)
    dataset.set_format("torch")

    return dataset, num_labels


def evaluate_vision(model, dataset, device):
    """Evaluate — only pass pixel_values and labels (no input_ids)"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            # Explicitly only pass pixel_values and labels
            inputs = {
                "pixel_values": item["pixel_values"].unsqueeze(0).to(device),
                "labels": item["labels"].unsqueeze(0).to(device),
            }
            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=1).item()
            if pred == inputs["labels"].item():
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def run_lora_sequential(seed, task_configs, processor):
    """LoRA sequential training on vision tasks"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Vision LoRA Sequential: Seed={seed}")
    logger.info(f"{'='*80}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    max_labels = max(t["num_labels"] for t in task_configs)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=max_labels, ignore_mismatched_sizes=True,
    )

    # FIX: No task_type — let PEFT auto-detect for vision model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        # NO task_type here — this was the bug
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"  ViT LoRA: {trainable:,} trainable / {total_p:,} total ({trainable/total_p*100:.2f}%)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load test sets
    val_datasets = {}
    for tc in task_configs:
        val_ds, _ = load_vision_dataset(tc["name"], processor, split="test", max_samples=2000)
        val_datasets[tc["name"]] = val_ds
        logger.info(f"  Loaded {tc['name']} test: {len(val_ds)} samples")

    accuracy_matrix = {tc["name"]: {} for tc in task_configs}

    for i, tc in enumerate(task_configs):
        current_task = tc["name"]
        logger.info(f"\n--- Training Task {i+1}/{len(task_configs)}: {current_task} ---")

        train_ds, _ = load_vision_dataset(current_task, processor, split="train", max_samples=5000)
        logger.info(f"  Training samples: {len(train_ds)}")

        training_args = TrainingArguments(
            output_dir=f"checkpoints/vision/lora_{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            fp16=True,
            learning_rate=2e-5,
            warmup_steps=50,
            weight_decay=0.01,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)

        start = time.time()
        trainer.train()
        logger.info(f"  Trained in {(time.time()-start)/60:.1f} min")

        # Evaluate
        for j, eval_tc in enumerate(task_configs[:i+1]):
            eval_task = eval_tc["name"]
            acc = evaluate_vision(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc

            if j < i:
                initial = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgot = initial - acc
                logger.info(f"    {eval_task}: {acc:.4f} (Forgot: {forgot:.4f})")
            else:
                logger.info(f"    {eval_task}: {acc:.4f} (just trained)")

        del trainer
        torch.cuda.empty_cache()

    # Forgetting
    last_task = task_configs[-1]["name"]
    forgetting_per_task = {}
    for tc in task_configs[:-1]:
        name = tc["name"]
        forgetting_per_task[name] = accuracy_matrix[name][f"after_{name}"] - accuracy_matrix[name][f"after_{last_task}"]

    return accuracy_matrix, {
        "avg_forgetting": float(np.mean(list(forgetting_per_task.values()))),
        "max_forgetting": float(np.max(list(forgetting_per_task.values()))),
        "forgetting_per_task": {k: round(v, 4) for k, v in forgetting_per_task.items()},
        "final_avg_accuracy": float(np.mean([accuracy_matrix[tc["name"]][f"after_{last_task}"] for tc in task_configs])),
        "initial_avg_accuracy": float(np.mean([accuracy_matrix[tc["name"]][f"after_{tc['name']}"] for tc in task_configs])),
    }


def main():
    logger.info("=" * 80)
    logger.info("VISION LORA RERUN (TaskType bug fixed)")
    logger.info("=" * 80)

    results_dir = Path("results/vision_vit")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    task_configs = [
        {"name": "cifar10", "num_labels": 10},
        {"name": "svhn", "num_labels": 10},
        {"name": "cifar100", "num_labels": 20},
    ]
    seeds = [42, 43, 44]

    # Load existing Full FT results
    all_results = {}
    for seed in seeds:
        ft_file = results_dir / f"vit_full_seed{seed}.json"
        if ft_file.exists():
            with open(ft_file) as f:
                all_results[f"full_seed{seed}"] = json.load(f)
            logger.info(f"  Loaded existing Full FT seed {seed}")

    start_time = time.time()

    for seed in seeds:
        exp_key = f"lora_seed{seed}"
        logger.info(f"\n{'#'*80}")
        logger.info(f"Vision LoRA: {exp_key}")
        logger.info(f"{'#'*80}")

        try:
            exp_start = time.time()
            accuracy_matrix, metrics = run_lora_sequential(seed, task_configs, processor)
            exp_time = (time.time() - exp_start) / 3600

            result = {
                "model": MODEL_NAME,
                "method": "lora",
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "task_sequence": [t["name"] for t in task_configs],
                "accuracy_matrix": accuracy_matrix,
                "forgetting_metrics": {k: round(v, 4) if isinstance(v, float) else
                                       {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v
                                       for k, v in metrics.items()},
                "time_hours": round(exp_time, 4),
            }
            all_results[exp_key] = result

            with open(results_dir / f"vit_{exp_key}.json", 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"\n✅ {exp_key}: Forgetting={metrics['avg_forgetting']:.4f}, "
                       f"Accuracy={metrics['final_avg_accuracy']:.4f}")

        except Exception as e:
            logger.error(f"❌ {exp_key} failed: {e}")
            logger.error(traceback.format_exc())

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = (time.time() - start_time) / 3600

    # Build combined summary
    method_summaries = {}
    for method in ["full", "lora"]:
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
        "experiment": "Vision ViT Cross-Modality (Combined)",
        "model": MODEL_NAME,
        "lora_rerun_time_hours": round(total_time, 2),
        "successful": len(all_results),
        "method_summary": method_summaries,
        "results": all_results,
    }

    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n\n" + "=" * 70)
    print("VISION (ViT) COMPLETE RESULTS")
    print("=" * 70)
    for method, stats in method_summaries.items():
        print(f"{method:<15} Forgetting: {stats['avg_forgetting_mean']:.4f} ± {stats['avg_forgetting_std']:.4f}  "
              f"Acc: {stats['final_accuracy_mean']:.4f} ± {stats['final_accuracy_std']:.4f}")
    print("-" * 70)
    if "full" in method_summaries and "lora" in method_summaries:
        ft_f = method_summaries["full"]["avg_forgetting_mean"]
        lora_f = method_summaries["lora"]["avg_forgetting_mean"]
        reduction = (ft_f - lora_f) / ft_f * 100 if ft_f > 0 else 0
        print(f"Forgetting reduction: {reduction:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
