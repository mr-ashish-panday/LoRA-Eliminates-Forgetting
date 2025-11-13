"""
Phase 8: DISTRIBUTION SHIFT ROBUSTNESS
Test best combination (all_4bit) on out-of-distribution tasks
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
    """Create model with best combination: all_linear + 4-bit"""
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


def run_phase8():
    logger.info("=" * 100)
    logger.info("PHASE 8: DISTRIBUTION SHIFT ROBUSTNESS")
    logger.info("=" * 100)
    
    results_dir = Path("results/phase8")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    train_tasks = ["mrpc", "cola", "sst2"]
    test_tasks = ["rte", "mrpc", "cola"]
    seeds = [42, 43]
    
    all_results = {}
    completed = 0
    total = len(train_tasks) * len(test_tasks) * len(seeds)
    start_time_overall = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    for train_task in train_tasks:
        for test_task in test_tasks:
            if train_task == test_task:
                continue
                
            for seed in seeds:
                completed += 1
                logger.info(f"\n[{completed}/{total}] Train: {train_task}, Test: {test_task}, Seed: {seed}")
                
                try:
                    experiment_start = time.time()
                    
                    model = create_best_model(seed)
                    
                    train_dataset = load_dataset("glue", train_task, split="train")
                    test_dataset = load_dataset("glue", test_task, split="validation[:200]")
                    
                    def tokenize(batch):
                        if "sentence2" in batch:
                            texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
                        else:
                            texts = batch["sentence"]
                        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
                    
                    train_dataset = train_dataset.rename_column("label", "labels")
                    test_dataset = test_dataset.rename_column("label", "labels")
                    
                    train_cols = [col for col in train_dataset.column_names if col != "labels"]
                    test_cols = [col for col in test_dataset.column_names if col != "labels"]
                    
                    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_cols)
                    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=test_cols)
                    
                    train_dataset.set_format("torch")
                    test_dataset.set_format("torch")
                    
                    training_args = TrainingArguments(
                        output_dir=f"checkpoints/phase8/{train_task}_to_{test_task}_seed{seed}",
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
                    
                    torch.cuda.reset_peak_memory_stats()
                    train_result = trainer.train()
                    
                    model.eval()
                    correct = 0
                    device = next(model.parameters()).device
                    
                    with torch.no_grad():
                        for i in range(len(test_dataset)):
                            sample = {k: v.unsqueeze(0).to(device) for k, v in test_dataset[i].items()}
                            outputs = model(**sample)
                            if outputs.logits.argmax(dim=1).item() == sample["labels"].item():
                                correct += 1
                    
                    accuracy = correct / len(test_dataset)
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    experiment_time = (time.time() - experiment_start) / 3600
                    
                    result = {
                        "experiment_id": f"phase8_{train_task}_to_{test_task}_seed{seed}",
                        "train_task": train_task,
                        "test_task": test_task,
                        "seed": seed,
                        "metrics": {
                            "accuracy": round(accuracy, 4),
                            "training_time_hours": round(experiment_time, 4),
                            "peak_gpu_memory_gb": round(peak_memory, 2),
                        },
                    }
                    
                    all_results[f"{train_task}_to_{test_task}_seed{seed}"] = result
                    
                    with open(results_dir / f"phase8_{train_task}_to_{test_task}_seed{seed}.json", 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    logger.info(f"  ✅ acc={accuracy:.3f}, mem={peak_memory:.2f}GB")
                    
                    del model, trainer
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
    
    total_time = (time.time() - start_time_overall) / 3600
    
    summary = {
        "phase": 8,
        "total_experiments": len(all_results),
        "total_time_hours": round(total_time, 2),
        "results": all_results,
    }
    
    with open(results_dir / "phase8_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 8 COMPLETED! Time: {total_time:.2f}h")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase8()
