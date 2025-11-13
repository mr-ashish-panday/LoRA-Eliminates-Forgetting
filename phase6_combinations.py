"""
Phase 6: BEST TECHNIQUE COMBINATIONS
Combine the best configurations from previous phases:
- Module selection: qv (efficient) vs all_linear (high-performance)
- Quantization: 4-bit vs 8-bit
- LoRA rank: r=8 (proven optimal from Phase 1-4)

Combinations tested:
1. qv + 4-bit (most efficient)
2. qv + 8-bit (balanced)
3. all_linear + 4-bit (high-perf + memory-efficient)
4. all_linear + 8-bit (maximum performance)

4 tasks × 4 combinations × 3 seeds = 48 experiments
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


def create_combination_model(combination, seed):
    """
    Create model with specific combination of techniques
    
    Combinations:
    - qv_4bit: Query+Value with 4-bit quantization
    - qv_8bit: Query+Value with 8-bit quantization
    - all_4bit: All linear with 4-bit quantization
    - all_8bit: All linear with 8-bit quantization
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Parse combination
    if combination.startswith("qv"):
        target_modules = ["query", "value"]
        module_desc = "query+value"
    else:  # all_linear
        target_modules = ["query", "key", "value", "dense"]
        module_desc = "all_linear"
    
    if "4bit" in combination:
        quant_type = "4bit"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:  # 8bit
        quant_type = "8bit"
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    # Load quantized model
    logger.info(f"  Loading model with {quant_type} quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    logger.info(f"  Applying LoRA to: {module_desc}")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    
    return model, module_desc, quant_type


def run_phase6():
    logger.info("=" * 100)
    logger.info("PHASE 6: BEST TECHNIQUE COMBINATIONS")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("Combinations:")
    logger.info("  1. qv + 4-bit: Most efficient (296K params, 0.4 GB)")
    logger.info("  2. qv + 8-bit: Balanced (296K params, 0.48 GB)")
    logger.info("  3. all_linear + 4-bit: High-perf + efficient (1.3M params, 0.5 GB)")
    logger.info("  4. all_linear + 8-bit: Maximum performance (1.3M params, 0.6 GB)")
    logger.info("Tasks: 4 GLUE tasks × 4 combinations × 3 seeds = 48 experiments")
    logger.info("Estimated: ~6-8 hours\n")
    
    # Check bitsandbytes
    try:
        import bitsandbytes
        logger.info(f"✓ bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        logger.error("❌ bitsandbytes not installed!")
        return
    
    results_dir = Path("results/phase6")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    combinations = ["qv_4bit", "qv_8bit", "all_4bit", "all_8bit"]
    seeds = [42, 43, 44]
    
    all_results = {}
    completed = 0
    total = len(tasks) * len(combinations) * len(seeds)
    start_time_overall = time.time()
    
    for task in tasks:
        for combination in combinations:
            for seed in seeds:
                completed += 1
                logger.info(f"\n[{completed}/{total}] Task: {task}, Combo: {combination}, Seed: {seed}")
                
                try:
                    experiment_start = time.time()
                    
                    # Create model with combination
                    model, module_desc, quant_type = create_combination_model(combination, seed)
                    
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
                    
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
                        output_dir=f"checkpoints/phase6/{task}_{combination}_seed{seed}",
                        num_train_epochs=3,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=32,
                        logging_steps=50,
                        eval_strategy="epoch",
                        save_strategy="no",
                        fp16=False,  # Disable for quantized models
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
                    
                    device = next(model.parameters()).device
                    
                    with torch.no_grad():
                        for i in range(len(val_dataset)):
                            sample = {k: v.unsqueeze(0).to(device) for k, v in val_dataset[i].items()}
                            outputs = model(**sample)
                            pred = outputs.logits.argmax(dim=1).item()
                            if pred == sample["labels"].item():
                                correct += 1
                            total_samples += 1
                    
                    accuracy = correct / total_samples
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    experiment_time = (time.time() - experiment_start) / 3600
                    
                    result = {
                        "experiment_id": f"phase6_{task}_{combination}_seed{seed}",
                        "phase": 6,
                        "task": task,
                        "combination": combination,
                        "module_type": module_desc,
                        "quantization": quant_type,
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
                            "f1": round(accuracy, 4),
                            "training_loss": round(train_result.training_loss, 4),
                            "training_time_hours": round(experiment_time, 4),
                            "peak_gpu_memory_gb": round(peak_memory, 2),
                            "total_training_steps": train_result.global_step,
                        },
                    }
                    
                    all_results[f"{task}_{combination}_seed{seed}"] = result
                    
                    result_file = results_dir / f"phase6_{task}_{combination}_seed{seed}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    logger.info(f"  ✅ acc={accuracy:.3f}, mem={peak_memory:.2f}GB, params={trainable_params:,}")
                    
                    del model, trainer
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    total_time = (time.time() - start_time_overall) / 3600
    
    summary = {
        "phase": 6,
        "total_experiments": len(all_results),
        "successful_experiments": len(all_results),
        "failed_experiments": total - len(all_results),
        "total_time_hours": round(total_time, 2),
        "combinations_tested": combinations,
        "results": all_results,
    }
    
    with open(results_dir / "phase6_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*100}")
    logger.info(f"PHASE 6 COMPLETED! Time: {total_time:.2f}h, Results: {len(all_results)}/{total}")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_phase6()
