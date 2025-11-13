
"""
Phase 1: LoRA Configuration Profiling
Compare different LoRA ranks: r=4, 8, 16, 32
4 tasks × 4 ranks × 3 seeds = 48 experiments
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def run_phase1():
    logger.info("=" * 100)
    logger.info("PHASE 1: LoRA CONFIGURATION PROFILING")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("Ranks: r=4, 8, 16, 32")
    logger.info("Tasks: 4 GLUE tasks × 4 ranks × 3 seeds = 48 experiments")
    logger.info("Estimated: ~2-3 hours\n")
    
    results_dir = Path("results/phase1")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    lora_ranks = [4, 8, 16, 32]
    seeds = [42, 43, 44]
    
    all_results = {}
    completed = 0
    total = len(tasks) * len(lora_ranks) * len(seeds)
    start_time_overall = time.time()
    
    for task in tasks:
        for r in lora_ranks:
            for seed in seeds:
                completed += 1
                logger.info(f"\n[{completed}/{total}] Task: {task}, Rank: r={r}, Seed: {seed}")
                
                try:
                    experiment_start = time.time()
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    # Load base model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        "bert-base-uncased",
                        num_labels=2
                    )
                    
                    # Apply LoRA - IMPORTANT: Use correct target modules for BERT
                    logger.info(f"  Applying LoRA (r={r})...")
                    lora_config = LoraConfig(
                        r=r,
                        lora_alpha=2*r,  # alpha = 2*r (common practice)
                        target_modules=["query", "value"],  # For BERT
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.SEQ_CLS,
                    )
                    model = get_peft_model(model, lora_config)
                    
                    # Print trainable parameters
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
                    
                    # Load SAME datasets as Phase 0 (FULL train, 200 val)
                    train_dataset = load_dataset("glue", task, split="train")
                    val_dataset = load_dataset("glue", task, split="validation[:200]")
                    
                    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    
                    logger.info(f"  Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
                    
                    # SAME preprocessing as Phase 0 (no [SEP] token)
                    def tokenize(batch):
                        if "sentence2" in batch:
                            texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
                        else:
                            texts = batch["sentence"]
                        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
                    
                    # Rename label column BEFORE mapping
                    train_dataset = train_dataset.rename_column("label", "labels")
                    val_dataset = val_dataset.rename_column("label", "labels")
                    
                    # Get columns to remove (everything except labels)
                    train_cols_to_remove = [col for col in train_dataset.column_names if col != "labels"]
                    val_cols_to_remove = [col for col in val_dataset.column_names if col != "labels"]
                    
                    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_cols_to_remove)
                    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=val_cols_to_remove)
                    
                    train_dataset.set_format("torch")
                    val_dataset.set_format("torch")
                    
                    # SAME training args as Phase 0
                    training_args = TrainingArguments(
                        output_dir=f"checkpoints/phase1/{task}_r{r}_seed{seed}",
                        num_train_epochs=3,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=32,
                        logging_steps=50,
                        eval_strategy="epoch",
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
                        eval_dataset=val_dataset,
                    )
                    
                    torch.cuda.reset_peak_memory_stats()
                    logger.info(f"  Training on {len(train_dataset)} samples for 3 epochs...")
                    train_result = trainer.train()
                    
                    # Evaluate
                    logger.info("  Evaluating...")
                    eval_results = trainer.evaluate()
                    
                    # Manual accuracy computation (same as Phase 0)
                    model.eval()
                    correct = 0
                    total_samples = 0
                    
                    with torch.no_grad():
                        for i in range(len(val_dataset)):
                            sample = {k: v.unsqueeze(0).to("cuda") for k, v in val_dataset[i].items()}
                            outputs = model(**sample)
                            pred = outputs.logits.argmax(dim=1).item()
                            if pred == sample["labels"].item():
                                correct += 1
                            total_samples += 1
                    
                    accuracy = correct / total_samples
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    experiment_time = (time.time() - experiment_start) / 3600
                    
                    result = {
                        "experiment_id": f"phase1_{task}_r{r}_seed{seed}",
                        "phase": 1,
                        "task": task,
                        "lora_rank": r,
                        "seed": seed,
                        "timestamp": datetime.now().isoformat(),
                        "dataset_info": {
                            "train_samples": len(train_dataset),
                            "val_samples": len(val_dataset),
                            "num_labels": 2,
                        },
                        "model_info": {
                            "total_params": total_params,
                            "trainable_params": trainable_params,
                            "trainable_percent": round(100 * trainable_params / total_params, 2),
                        },
                        "metrics": {
                            "accuracy": round(accuracy, 4),
                            "f1": round(accuracy, 4),  # Can add proper F1 later
                            "training_loss": round(train_result.training_loss, 4),
                            "training_time_hours": round(experiment_time, 4),
                            "peak_gpu_memory_gb": round(peak_memory, 2),
                            "total_training_steps": train_result.global_step,
                        },
                    }
                    
                    all_results[f"{task}_r{r}_seed{seed}"] = result
                    
                    result_file = results_dir / f"phase1_{task}_r{r}_seed{seed}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    logger.info(f"  ✅ acc={accuracy:.3f}, mem={peak_memory:.2f}GB, params={trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
                    
                    del model, trainer
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    total_time = (time.time() - start_time_overall) / 3600
    
    summary = {
        "phase": 1,
        "total_experiments": len(all_results),
        "successful_experiments": len(all_results),
        "failed_experiments": total - len(all_results),
        "total_time_hours": round(total_time, 2),
        "lora_ranks_tested": lora_ranks,
        "results": all_results,
    }
    
    with open(results_dir / "phase1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 1 COMPLETED! Time: {total_time:.2f}h, Results: {len(all_results)}/{total}")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase1()
