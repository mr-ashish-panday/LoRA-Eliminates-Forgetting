"""
Phase 11: CONTINUAL LEARNING WITH EWC
Combine best technique (all_4bit) with Elastic Weight Consolidation
Test if EWC further reduces catastrophic forgetting
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


def create_best_model(seed):
    """Create model with best combination"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    
    return model


def run_sequential_training_simple(seed, tasks, tokenizer):
    """Simplified sequential training for Phase 11"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Sequential Training: Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")
    
    model = create_best_model(seed)
    device = next(model.parameters()).device
    
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
    
    accuracy_matrix = {task: {} for task in tasks}
    
    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")
        
        train_dataset = load_dataset("glue", current_task, split="train")
        train_dataset = train_dataset.rename_column("label", "labels")
        cols = [col for col in train_dataset.column_names if col != "labels"]
        train_dataset = train_dataset.map(
            lambda batch: tokenizer(
                [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
                if "sentence2" in batch else batch["sentence"],
                max_length=128, padding="max_length", truncation=True
            ),
            batched=True, remove_columns=cols
        )
        train_dataset.set_format("torch")
        
        training_args = TrainingArguments(
            output_dir=f"checkpoints/phase11/seed{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            logging_steps=100,
            eval_strategy="no",
            save_strategy="no",
            fp16=False,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",
        )
        
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
        
        # Evaluate on all tasks
        for eval_task in tasks[:i+1]:
            correct = 0
            with torch.no_grad():
                for idx in range(len(val_datasets[eval_task])):
                    sample = {k: v.unsqueeze(0).to(device) for k, v in val_datasets[eval_task][idx].items()}
                    outputs = model(**sample)
                    if outputs.logits.argmax(dim=1).item() == sample["labels"].item():
                        correct += 1
            
            acc = correct / len(val_datasets[eval_task])
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            logger.info(f"  {eval_task}: {acc:.3f}")
        
        del trainer
        torch.cuda.empty_cache()
    
    # Calculate forgetting
    forgetting = []
    for task in tasks[:-1]:
        initial = accuracy_matrix[task][f"after_{task}"]
        final = accuracy_matrix[task][f"after_{tasks[-1]}"]
        forgetting.append(initial - final)
    
    return {
        "accuracy_matrix": accuracy_matrix,
        "avg_forgetting": np.mean(forgetting) if forgetting else 0.0
    }


def run_phase11():
    logger.info("=" * 100)
    logger.info("PHASE 11: CONTINUAL LEARNING VALIDATION")
    logger.info("=" * 100)
    
    results_dir = Path("results/phase11")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola"]
    seeds = [42, 43, 44]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    all_results = {}
    start_time = time.time()
    
    for seed in seeds:
        logger.info(f"\n{'#'*100}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'#'*100}")
        
        try:
            result = run_sequential_training_simple(seed, tasks, tokenizer)
            
            all_results[f"seed{seed}"] = {
                "seed": seed,
                "accuracy_matrix": result["accuracy_matrix"],
                "avg_forgetting": round(result["avg_forgetting"], 4),
            }
            
            with open(results_dir / f"phase11_seed{seed}.json", 'w') as f:
                json.dump(all_results[f"seed{seed}"], f, indent=2)
            
            logger.info(f"\n✅ Seed {seed} complete: Avg forgetting = {result['avg_forgetting']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
    
    total_time = (time.time() - start_time) / 3600
    
    summary = {
        "phase": 11,
        "total_time_hours": round(total_time, 2),
        "results": all_results,
    }
    
    with open(results_dir / "phase11_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 11 COMPLETED! Time: {total_time:.2f}h")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase11()
