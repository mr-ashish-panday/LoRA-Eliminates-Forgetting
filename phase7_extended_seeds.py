"""
Phase 7 Extended: Add 5 More Seeds for Statistical Rigor
Original Phase 7 had seeds 42, 43, 44
Now adding: 45, 46, 47, 48, 49
Total: 8 seeds for robust statistics

3 methods × 5 new seeds = 15 experiments
Time: ~2 hours
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time

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
    
    elif method == "4bit":
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


def evaluate_on_task(model, dataset, device):
    """Evaluate model on a task dataset"""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**sample)
            pred = outputs.logits.argmax(dim=1).item()
            if pred == sample["labels"].item():
                correct += 1
    
    return correct / len(dataset)


def run_sequential_training(method, seed, tasks, tokenizer):
    """Train on tasks sequentially and measure forgetting"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Sequential Training: Method={method}, Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")
    
    model, method_name = create_model(method, seed)
    device = next(model.parameters()).device
    
    # Load all validation sets
    val_datasets = {}
    for task in tasks:
        val_dataset = load_dataset("glue", task, split="validation[:200]")
        
        def tokenize(batch):
            if "sentence2" in batch:
                texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
            else:
                texts = batch["sentence"]
            return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
        
        val_dataset = val_dataset.rename_column("label", "labels")
        cols = [col for col in val_dataset.column_names if col != "labels"]
        val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=cols)
        val_dataset.set_format("torch")
        val_datasets[task] = val_dataset
        logger.info(f"  Loaded {task} validation: {len(val_dataset)} samples")
    
    accuracy_matrix = {task: {} for task in tasks}
    
    # Train each task sequentially
    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")
        
        train_dataset = load_dataset("glue", current_task, split="train")
        
        # Limit SST2 for speed
        if current_task == "sst2" and len(train_dataset) > 2000:
            indices = np.random.choice(len(train_dataset), 2000, replace=False)
            train_dataset = train_dataset.select(indices)
        
        logger.info(f"  Training samples: {len(train_dataset)}")
        
        def tokenize(batch):
            if "sentence2" in batch:
                texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
            else:
                texts = batch["sentence"]
            return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
        
        train_dataset = train_dataset.rename_column("label", "labels")
        cols = [col for col in train_dataset.column_names if col != "labels"]
        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=cols)
        train_dataset.set_format("torch")
        
        training_args = TrainingArguments(
            output_dir=f"checkpoints/phase7_extended/{method}_{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            fp16=(method == "full" or method == "lora"),
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
        
        trainer.train()
        
        # Evaluate on all previous tasks
        logger.info(f"  Evaluating on all tasks:")
        for eval_task in tasks[:i+1]:
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            
            if eval_task != current_task:
                initial_acc = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgetting = initial_acc - acc
                logger.info(f"    {eval_task}: {acc:.3f} (Initial: {initial_acc:.3f}, Forgot: {forgetting:.3f})")
            else:
                logger.info(f"    {eval_task}: {acc:.3f} (just trained)")
        
        del trainer
        torch.cuda.empty_cache()
    
    # Calculate forgetting metrics
    forgetting_per_task = {}
    for task in tasks[:-1]:
        initial = accuracy_matrix[task][f"after_{task}"]
        final = accuracy_matrix[task][f"after_{tasks[-1]}"]
        forgetting_per_task[task] = initial - final
    
    avg_forgetting = np.mean(list(forgetting_per_task.values()))
    max_forgetting = np.max(list(forgetting_per_task.values()))
    
    return {
        "accuracy_matrix": accuracy_matrix,
        "forgetting_per_task": forgetting_per_task,
        "avg_forgetting": avg_forgetting,
        "max_forgetting": max_forgetting,
    }


def run_phase7_extended():
    logger.info("=" * 100)
    logger.info("PHASE 7 EXTENDED: ADDITIONAL SEEDS FOR STATISTICAL RIGOR")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("Adding seeds: 45, 46, 47, 48, 49")
    logger.info("Methods: Full fine-tuning, LoRA r=8, 4-bit+LoRA")
    logger.info("3 methods × 5 seeds = 15 experiments\n")
    
    results_dir = Path("results/phase7_extended")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    methods = ["full", "lora", "4bit"]
    seeds = [45, 46, 47, 48, 49]  # NEW SEEDS
    
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
                
                result = run_sequential_training(method, seed, tasks, tokenizer)
                
                experiment_time = (time.time() - experiment_start) / 3600
                
                result_data = {
                    "experiment_id": f"phase7_extended_{method}_seed{seed}",
                    "phase": 7,
                    "extension": True,
                    "method": method,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "task_sequence": tasks,
                    "accuracy_matrix": result["accuracy_matrix"],
                    "forgetting_metrics": {
                        "avg_forgetting": round(result["avg_forgetting"], 4),
                        "max_forgetting": round(result["max_forgetting"], 4),
                        "forgetting_per_task": {k: round(v, 4) for k, v in result["forgetting_per_task"].items()},
                    },
                    "metrics": {
                        "total_time_hours": round(experiment_time, 4),
                    }
                }
                
                all_results[f"{method}_seed{seed}"] = result_data
                
                with open(results_dir / f"phase7_extended_{method}_seed{seed}.json", 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                logger.info(f"\n✅ EXPERIMENT COMPLETED!")
                logger.info(f"   Avg Forgetting: {result['avg_forgetting']:.3f}")
                logger.info(f"   Max Forgetting: {result['max_forgetting']:.3f}")
                logger.info(f"   Time: {experiment_time:.2f}h")
                
            except Exception as e:
                logger.error(f"❌ Error in experiment: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    total_time = (time.time() - start_time_overall) / 3600
    
    summary = {
        "phase": 7,
        "extension": True,
        "total_experiments": len(all_results),
        "successful_experiments": len(all_results),
        "failed_experiments": total - len(all_results),
        "total_time_hours": round(total_time, 2),
        "new_seeds": seeds,
        "methods_tested": methods,
        "results": all_results,
    }
    
    with open(results_dir / "phase7_extended_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 7 EXTENDED COMPLETED!")
    logger.info(f"Total Time: {total_time:.2f}h")
    logger.info(f"Successful: {len(all_results)}/{total}")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase7_extended()
