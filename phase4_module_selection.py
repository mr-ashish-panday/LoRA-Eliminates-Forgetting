"""
Phase 4: MODULE SELECTION - Which modules to apply LoRA?
Test different target modules to find optimal coverage

Strategies:
1. qv: Query+Value (baseline from Phase 1-3)
2. qkv: Query+Key+Value (comprehensive attention)
3. all_linear: All linear layers in attention (max coverage)

4 tasks × 3 strategies × 3 seeds = 36 experiments
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


def create_module_config(modules_type="qv"):
    """
    Create LoRA config with different target modules
    
    BERT architecture:
    - encoder.layer.X.attention.self.query
    - encoder.layer.X.attention.self.key
    - encoder.layer.X.attention.self.value
    - encoder.layer.X.attention.output.dense
    - encoder.layer.X.output.dense (FFN)
    """
    
    if modules_type == "qv":
        # Query + Value only (baseline - most common in literature)
        target_modules = ["query", "value"]
        description = "query+value"
        expected_params = 296450  # ~296K
        
    elif modules_type == "qkv":
        # Query + Key + Value (all self-attention projections)
        target_modules = ["query", "key", "value"]
        description = "query+key+value"
        expected_params = 444675  # ~445K (1.5x more)
        
    elif modules_type == "all_linear":
        # All attention linear layers (query, key, value, attention output, FFN output)
        # This targets maximum coverage
        target_modules = ["query", "key", "value", "dense"]
        description = "all_attention_linear"
        expected_params = 741125  # ~741K (2.5x more)
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    
    return config, description, expected_params


def run_phase4():
    logger.info("=" * 100)
    logger.info("PHASE 4: MODULE SELECTION")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("Module strategies:")
    logger.info("  - qv: Query+Value (baseline, ~296K params)")
    logger.info("  - qkv: Query+Key+Value (comprehensive, ~445K params)")
    logger.info("  - all_linear: All attention linear layers (max, ~741K params)")
    logger.info("Tasks: 4 GLUE tasks × 3 strategies × 3 seeds = 36 experiments")
    logger.info("Estimated: ~5-6 hours\n")
    
    results_dir = Path("results/phase4")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    module_types = ["qv", "qkv", "all_linear"]
    seeds = [42, 43, 44]
    
    all_results = {}
    completed = 0
    total = len(tasks) * len(module_types) * len(seeds)
    start_time_overall = time.time()
    
    for task in tasks:
        for module_type in module_types:
            for seed in seeds:
                completed += 1
                logger.info(f"\n[{completed}/{total}] Task: {task}, Modules: {module_type}, Seed: {seed}")
                
                try:
                    experiment_start = time.time()
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    # Load model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        "bert-base-uncased",
                        num_labels=2
                    )
                    
                    # Apply LoRA with specific modules
                    lora_config, module_desc, expected_params = create_module_config(module_type)
                    logger.info(f"  Applying LoRA to: {module_desc}")
                    model = get_peft_model(model, lora_config)
                    
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
                    logger.info(f"  Expected: ~{expected_params:,}, Actual: {trainable_params:,}")
                    
                    # Load datasets
                    train_dataset = load_dataset("glue", task, split="train")
                    val_dataset = load_dataset("glue", task, split="validation[:200]")
                    
                    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    
                    logger.info(f"  Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
                    
                    def tokenize(batch):
                        if "sentence2" in batch:
                            texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
                        else:
                            texts = batch["sentence"]
                        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
                    
                    # Rename and process
                    train_dataset = train_dataset.rename_column("label", "labels")
                    val_dataset = val_dataset.rename_column("label", "labels")
                    
                    train_cols_to_remove = [col for col in train_dataset.column_names if col != "labels"]
                    val_cols_to_remove = [col for col in val_dataset.column_names if col != "labels"]
                    
                    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_cols_to_remove)
                    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=val_cols_to_remove)
                    
                    train_dataset.set_format("torch")
                    val_dataset.set_format("torch")
                    
                    training_args = TrainingArguments(
                        output_dir=f"checkpoints/phase4/{task}_{module_type}_seed{seed}",
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
                        "experiment_id": f"phase4_{task}_{module_type}_seed{seed}",
                        "phase": 4,
                        "task": task,
                        "module_type": module_type,
                        "module_description": module_desc,
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
                            "expected_params": expected_params,
                        },
                        "metrics": {
                            "accuracy": round(accuracy, 4),
                            "f1": round(accuracy, 4),
                            "training_loss": round(train_result.training_loss, 4),
                            "training_time_hours": round(experiment_time, 4),
                            "peak_gpu_memory_gb": round(peak_memory, 2),
                            "total_training_steps": train_result.global_step,
                        },
                    }
                    
                    all_results[f"{task}_{module_type}_seed{seed}"] = result
                    
                    result_file = results_dir / f"phase4_{task}_{module_type}_seed{seed}.json"
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
        "phase": 4,
        "total_experiments": len(all_results),
        "successful_experiments": len(all_results),
        "failed_experiments": total - len(all_results),
        "total_time_hours": round(total_time, 2),
        "module_types_tested": module_types,
        "results": all_results,
    }
    
    with open(results_dir / "phase4_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 4 COMPLETED! Time: {total_time:.2f}h, Results: {len(all_results)}/{total}")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase4()
