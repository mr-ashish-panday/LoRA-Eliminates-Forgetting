"""
Experiment 6: Vision (ViT + CIFAR) Cross-Modality Generalization
=================================================================
Proves LoRA's forgetting resistance works across modalities.
Sequential training on vision classification tasks using ViT-base.

Sequence: CIFAR-10 → SVHN → CIFAR-100 (coarse, 20 superclasses)
Full FT vs LoRA r=8, 3 seeds each.

Infrastructure notes:
- Uses AutoImageProcessor (NOT tokenizer) for ViT
- ViT attention modules: "query" and "value" (same names as BERT)
- save_strategy="no" to prevent disk bloat
- Each task capped at 5000 training samples for speed

Estimated time: ~6 hours on consumer GPU
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
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/vision_vit.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "google/vit-base-patch16-224-in21k"


# ========== DATA LOADING ==========

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
        num_labels = 20  # coarse labels (superclasses)
        image_key = "img"
        label_key = "coarse_label"

    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)

    def preprocess(batch):
        images = batch[image_key]
        # Convert to RGB if needed
        processed_images = []
        for img in images:
            if hasattr(img, 'convert'):
                img = img.convert("RGB")
            processed_images.append(img)

        inputs = processor(images=processed_images, return_tensors="pt")
        inputs["labels"] = torch.tensor(batch[label_key])
        return inputs

    # Process in batches
    dataset = dataset.rename_column(label_key, "labels") if label_key != "labels" else dataset

    def transform(batch):
        images = batch[image_key]
        processed = []
        for img in images:
            if hasattr(img, 'convert'):
                img = img.convert("RGB")
            processed.append(img)
        result = processor(images=processed, return_tensors="pt")
        return {
            "pixel_values": result["pixel_values"],
        }

    cols_to_remove = [c for c in dataset.column_names if c not in ["labels"]]
    dataset = dataset.map(transform, batched=True, batch_size=32, remove_columns=cols_to_remove)
    dataset.set_format("torch")

    return dataset, num_labels


# ========== EVALUATION ==========

def evaluate_vision(model, dataset, device):
    """Evaluate vision model accuracy"""
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

def create_vision_model(method, num_labels, seed):
    """Create ViT model with or without LoRA"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    if method == "lora":
        # ViT uses "query" and "value" in attention layers
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.IMAGE_CLASSIFICATION if hasattr(TaskType, 'IMAGE_CLASSIFICATION') else TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        logger.info(f"  ViT LoRA: {trainable:,} trainable / {total_p:,} total ({trainable/total_p*100:.2f}%)")

    return model


# ========== SEQUENTIAL TRAINING ==========

def run_vision_sequential(method, seed, task_configs, processor):
    """Sequential vision training with forgetting measurement"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Vision Sequential: Method={method}, Seed={seed}")
    logger.info(f"Tasks: {' → '.join([t['name'] for t in task_configs])}")
    logger.info(f"{'='*80}")

    task_names = [t["name"] for t in task_configs]

    # We need to handle changing num_labels between tasks
    # Strategy: Use the maximum num_labels and map outputs
    max_labels = max(t["num_labels"] for t in task_configs)

    model = create_vision_model(method, max_labels, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load all test sets
    val_datasets = {}
    task_num_labels = {}
    for tc in task_configs:
        name = tc["name"]
        val_ds, n_labels = load_vision_dataset(name, processor, split="test", max_samples=2000)
        val_datasets[name] = val_ds
        task_num_labels[name] = n_labels
        logger.info(f"  Loaded {name} test: {len(val_ds)} samples, {n_labels} classes")

    accuracy_matrix = {t["name"]: {} for t in task_configs}

    for i, tc in enumerate(task_configs):
        current_task = tc["name"]
        logger.info(f"\n--- Training Task {i+1}/{len(task_configs)}: {current_task} ---")

        train_ds, _ = load_vision_dataset(
            current_task, processor, split="train",
            max_samples=5000
        )
        logger.info(f"  Training samples: {len(train_ds)}")

        training_args = TrainingArguments(
            output_dir=f"checkpoints/vision/{method}_{seed}_{current_task}",
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

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        trainer.train()
        train_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"  Trained in {train_time/60:.1f} min, Peak: {peak_mem:.2f}GB")

        # Evaluate on all seen tasks
        logger.info(f"  Evaluating:")
        for j, eval_tc in enumerate(task_configs[:i+1]):
            eval_task = eval_tc["name"]
            acc = evaluate_vision(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc

            if j < i:
                initial = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgot = initial - acc
                logger.info(f"    {eval_task}: {acc:.4f} (Initial: {initial:.4f}, Forgot: {forgot:.4f})")
            else:
                logger.info(f"    {eval_task}: {acc:.4f} (just trained)")

        del trainer
        torch.cuda.empty_cache()

    # Calculate forgetting
    forgetting_per_task = {}
    for tc in task_configs[:-1]:
        name = tc["name"]
        initial = accuracy_matrix[name][f"after_{name}"]
        final = accuracy_matrix[name][f"after_{task_configs[-1]['name']}"]
        forgetting_per_task[name] = initial - final

    metrics = {
        "avg_forgetting": float(np.mean(list(forgetting_per_task.values()))),
        "max_forgetting": float(np.max(list(forgetting_per_task.values()))),
        "forgetting_per_task": {k: round(v, 4) for k, v in forgetting_per_task.items()},
        "final_avg_accuracy": float(np.mean([
            accuracy_matrix[tc["name"]][f"after_{task_configs[-1]['name']}"]
            for tc in task_configs
        ])),
        "initial_avg_accuracy": float(np.mean([
            accuracy_matrix[tc["name"]][f"after_{tc['name']}"]
            for tc in task_configs
        ])),
    }

    return accuracy_matrix, metrics


# ========== MAIN ==========

def run_vision_experiment():
    logger.info("=" * 100)
    logger.info("EXPERIMENT 6: Vision (ViT + CIFAR/SVHN)")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("=" * 100)

    results_dir = Path("results/vision_vit")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Verify ViT target modules before running
    logger.info("Verifying ViT architecture...")
    test_model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=10, ignore_mismatched_sizes=True
    )
    # Print layer names to verify target_modules
    vit_layers = [name for name, _ in test_model.named_modules()]
    query_layers = [n for n in vit_layers if "query" in n]
    value_layers = [n for n in vit_layers if "value" in n]
    logger.info(f"  Found {len(query_layers)} query layers, {len(value_layers)} value layers")
    if not query_layers or not value_layers:
        logger.error("  WARNING: 'query'/'value' not found! Listing all layers:")
        for name in vit_layers:
            if "attention" in name.lower():
                logger.info(f"    {name}")
        raise ValueError("ViT target_modules mismatch — check layer names")
    del test_model
    gc.collect()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    task_configs = [
        {"name": "cifar10", "num_labels": 10},
        {"name": "svhn", "num_labels": 10},
        {"name": "cifar100", "num_labels": 20},
    ]

    methods = ["full", "lora"]
    seeds = [42, 43, 44]

    all_results = {}
    start_time = time.time()

    for method in methods:
        for seed in seeds:
            exp_key = f"{method}_seed{seed}"
            logger.info(f"\n{'#'*80}")
            logger.info(f"Vision: {exp_key}")
            logger.info(f"{'#'*80}")

            try:
                exp_start = time.time()
                accuracy_matrix, metrics = run_vision_sequential(
                    method, seed, task_configs, processor
                )
                exp_time = (time.time() - exp_start) / 3600

                result = {
                    "model": MODEL_NAME,
                    "method": method,
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

    # Summaries
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
        "experiment": "Vision ViT Cross-Modality",
        "model": MODEL_NAME,
        "total_time_hours": round(total_time, 2),
        "successful": len(all_results),
        "method_summary": method_summaries,
        "results": all_results,
    }

    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n\n" + "=" * 70)
    print("VISION (ViT) RESULTS")
    print("=" * 70)
    for method, stats in method_summaries.items():
        print(f"{method:<15} Forgetting: {stats['avg_forgetting_mean']:.4f} ± {stats['avg_forgetting_std']:.4f}  "
              f"Acc: {stats['final_accuracy_mean']:.4f} ± {stats['final_accuracy_std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_vision_experiment()
