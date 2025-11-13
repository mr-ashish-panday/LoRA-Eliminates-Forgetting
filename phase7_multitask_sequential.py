"""
Phase 7: Multi-Task Sequential Learning - CATASTROPHIC FORGETTING
Train on tasks sequentially and measure forgetting

Task sequence: RTE → MRPC → CoLA → SST2
Methods: Full fine-tuning, LoRA r=8, 4-bit+LoRA
3 methods × 3 seeds = 9 experiments

For each experiment:
- Train Task 1, evaluate on Task 1
- Train Task 2, evaluate on Task 1 and Task 2 (measure Task 1 forgetting)
- Train Task 3, evaluate on all previous tasks
- Train Task 4, evaluate on all previous tasks

Key metrics:
- Backward Transfer: How much Task N hurts Task N-1 performance
- Forward Transfer: Does Task N help Task N+1?
- Average Forgetting: Mean accuracy drop across all tasks
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time
import copy

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(task, tokenizer, split="train", max_samples=None):
    """Load and tokenize dataset"""
    if split == "train":
        dataset = load_dataset("glue", task, split="train")
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
    else:
        dataset = load_dataset("glue", task, split="validation[:200]")
    
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


def evaluate_on_task(model, dataset, device):
    """Evaluate model on a task dataset"""
    model.eval()
    correct = 0
    total = 0
    
    # Move model to device if not already
    model = model.to(device)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**sample)
            pred = outputs.logits.argmax(dim=1).item()
            if pred == sample["labels"].item():
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def create_model(method, seed):
    """Create model based on method type"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if method == "full":
        # Full fine-tuning (Phase 0 style)
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        return model, "full_finetuning"
    
    elif method == "lora":
        # LoRA r=8 (Phase 1 style)
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
    
    elif method == "4bit":
        # 4-bit quantization + LoRA (Phase 2 style)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        return model, "4bit_lora"


def run_sequential_training(method, seed, tasks, tokenizer):
    """
    Train on tasks sequentially and measure forgetting
    
    Returns:
    - accuracy_matrix: [task_i][after_task_j] = accuracy of task_i after training task_j
    - forgetting_metrics: backward transfer, forward transfer, etc.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Sequential Training: Method={method}, Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")
    
    # Create model
    model, method_name = create_model(method, seed)
    device = next(model.parameters()).device
    
    # Load all validation sets upfront
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
            max_samples=2000 if current_task == "sst2" else None  # Limit SST2 for speed
        )
        logger.info(f"  Training samples: {len(train_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"checkpoints/phase7/{method}_{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            fp16=(method == "full" or method == "lora"),  # No fp16 for 4bit
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
        
        logger.info(f"  Training completed in {train_time/60:.1f} minutes")
        
        # Evaluate on ALL previous tasks (including current)
        logger.info(f"  Evaluating on all tasks seen so far:")
        for j, eval_task in enumerate(tasks[:i+1]):
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            
            # Check for forgetting
            if j < i:  # Previous task
                initial_acc = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgetting = initial_acc - acc
                logger.info(f"    {eval_task}: {acc:.3f} (Initial: {initial_acc:.3f}, Forgot: {forgetting:.3f})")
            else:  # Current task
                logger.info(f"    {eval_task}: {acc:.3f} (just trained)")
        
        del trainer
        torch.cuda.empty_cache()
    
    # Calculate forgetting metrics
    forgetting_metrics = calculate_forgetting_metrics(accuracy_matrix, tasks)
    
    return accuracy_matrix, forgetting_metrics


def calculate_forgetting_metrics(accuracy_matrix, tasks):
    """
    Calculate comprehensive forgetting metrics
    """
    metrics = {}
    
    # 1. Backward Transfer (forgetting): How much did we forget?
    backward_transfer = []
    for i, task in enumerate(tasks[:-1]):  # All except last
        initial_acc = accuracy_matrix[task][f"after_{task}"]
        final_acc = accuracy_matrix[task][f"after_{tasks[-1]}"]
        forgetting = initial_acc - final_acc
        backward_transfer.append(forgetting)
    
    metrics["avg_forgetting"] = np.mean(backward_transfer) if backward_transfer else 0.0
    metrics["max_forgetting"] = np.max(backward_transfer) if backward_transfer else 0.0
    metrics["forgetting_per_task"] = {tasks[i]: backward_transfer[i] for i in range(len(backward_transfer))}
    
    # 2. Final average accuracy
    final_accuracies = [accuracy_matrix[task][f"after_{tasks[-1]}"] for task in tasks]
    metrics["final_avg_accuracy"] = np.mean(final_accuracies)
    
    # 3. Initial average accuracy (right after training each task)
    initial_accuracies = [accuracy_matrix[task][f"after_{task}"] for task in tasks]
    metrics["initial_avg_accuracy"] = np.mean(initial_accuracies)
    
    return metrics


def run_phase7():
    logger.info("=" * 100)
    logger.info("PHASE 7: MULTI-TASK SEQUENTIAL LEARNING - CATASTROPHIC FORGETTING")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("Methods: Full fine-tuning, LoRA r=8, 4-bit+LoRA")
    logger.info("Task Sequence: RTE → MRPC → CoLA → SST2")
    logger.info("Experiments: 3 methods × 3 seeds = 9 experiments\n")
    
    results_dir = Path("results/phase7")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    methods = ["full", "lora", "4bit"]
    seeds = [42, 43, 44]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    all_results = {}
    completed = 0
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
                
                # Run sequential training
                accuracy_matrix, forgetting_metrics = run_sequential_training(
                    method, seed, tasks, tokenizer
                )
                
                experiment_time = (time.time() - experiment_start) / 3600
                
                # Save results
                result = {
                    "experiment_id": f"phase7_{method}_seed{seed}",
                    "phase": 7,
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
                logger.info(f"   Avg Forgetting: {forgetting_metrics['avg_forgetting']:.3f}")
                logger.info(f"   Max Forgetting: {forgetting_metrics['max_forgetting']:.3f}")
                logger.info(f"   Final Avg Acc: {forgetting_metrics['final_avg_accuracy']:.3f}")
                logger.info(f"   Time: {experiment_time:.2f}h")
                
            except Exception as e:
                logger.error(f"❌ Error in experiment: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    total_time = (time.time() - start_time_overall) / 3600
    
    # Save summary
    summary = {
        "phase": 7,
        "total_experiments": len(all_results),
        "successful_experiments": len(all_results),
        "failed_experiments": total - len(all_results),
        "total_time_hours": round(total_time, 2),
        "methods_tested": methods,
        "task_sequence": tasks,
        "results": all_results,
    }
    
    with open(results_dir / "phase7_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 7 COMPLETED!")
    logger.info(f"Total Time: {total_time:.2f}h")
    logger.info(f"Successful: {len(all_results)}/{total}")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase7()
