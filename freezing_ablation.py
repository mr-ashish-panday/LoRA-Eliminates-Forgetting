"""
FREEZING ABLATION STUDY
Test if frozen backbone is the key mechanism preventing forgetting

Configurations:
- 50% frozen (freeze layers 0-5, train 6-11)
- 75% frozen (freeze layers 0-8, train 9-11)  
- 95% frozen (freeze layers 0-10, train 11 only)
- 99.7% frozen (LoRA - already have data)

3 configs × 3 seeds × 4 tasks = 9 experiments (~2 hours)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def freeze_model_by_percentage(model, freeze_pct):
    """
    Freeze percentage of BERT layers
    BERT has 12 encoder layers (0-11)
    """
    total_layers = 12
    freeze_until = int(total_layers * freeze_pct)
    
    # Freeze embeddings
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze specified percentage of encoder layers
    for i in range(freeze_until):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
    
    # Rest are trainable
    for i in range(freeze_until, total_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = True
    
    # Classifier always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    logger.info(f"  Frozen: {freeze_pct*100:.0f}% of layers (0-{freeze_until-1})")
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return model


def evaluate_on_task(model, dataset, device):
    """Evaluate model on a task"""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**sample)
            if outputs.logits.argmax(dim=1).item() == sample["labels"].item():
                correct += 1
    
    return correct / len(dataset)


def run_sequential_training(freeze_pct, seed, tasks, tokenizer):
    """Run sequential training with specified freezing percentage"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Freezing: {freeze_pct*100:.0f}%, Seed: {seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    
    # Apply freezing
    model = freeze_model_by_percentage(model, freeze_pct)
    model = model.to("cuda")
    
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
    
    accuracy_matrix = {task: {} for task in tasks}
    
    # Train each task sequentially
    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")
        
        train_dataset = load_dataset("glue", current_task, split="train")
        train_dataset = train_dataset.rename_column("label", "labels")
        cols = [col for col in train_dataset.column_names if col != "labels"]
        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=cols)
        train_dataset.set_format("torch")
        
        training_args = TrainingArguments(
            output_dir=f"checkpoints/freezing_ablation/{int(freeze_pct*100)}pct_{seed}_{current_task}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            logging_steps=100,
            eval_strategy="no",
            save_strategy="no",
            fp16=True,
            learning_rate=2e-5,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",
        )
        
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
        
        # Evaluate on all previous tasks
        for eval_task in tasks[:i+1]:
            acc = evaluate_on_task(model, val_datasets[eval_task], "cuda")
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
        "avg_forgetting": np.mean(forgetting) if forgetting else 0.0,
        "forgetting_per_task": {tasks[i]: forgetting[i] for i in range(len(forgetting))}
    }


def run_ablation():
    logger.info("=" * 100)
    logger.info("FREEZING ABLATION STUDY")
    logger.info("=" * 100)
    
    results_dir = Path("results/freezing_ablation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    freeze_percentages = [0.50, 0.75, 0.95]  # 99.7% is LoRA (already have)
    seeds = [42, 43, 44]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    all_results = {}
    start_time = time.time()
    
    for freeze_pct in freeze_percentages:
        for seed in seeds:
            logger.info(f"\n{'#'*100}")
            logger.info(f"Configuration: {freeze_pct*100:.0f}% Frozen, Seed {seed}")
            logger.info(f"{'#'*100}")
            
            try:
                result = run_sequential_training(freeze_pct, seed, tasks, tokenizer)
                
                config_name = f"{int(freeze_pct*100)}pct_seed{seed}"
                all_results[config_name] = {
                    "freeze_percentage": freeze_pct,
                    "seed": seed,
                    "accuracy_matrix": result["accuracy_matrix"],
                    "avg_forgetting": round(result["avg_forgetting"], 4),
                    "forgetting_per_task": {k: round(v, 4) for k, v in result["forgetting_per_task"].items()}
                }
                
                with open(results_dir / f"{config_name}.json", 'w') as f:
                    json.dump(all_results[config_name], f, indent=2)
                
                logger.info(f"\n✅ Complete: Avg forgetting = {result['avg_forgetting']:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    total_time = (time.time() - start_time) / 3600
    
    summary = {
        "total_time_hours": round(total_time, 2),
        "configurations_tested": freeze_percentages,
        "results": all_results,
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"FREEZING ABLATION COMPLETED! Time: {total_time:.2f}h")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_ablation()
