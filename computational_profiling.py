"""
COMPUTATIONAL COST PROFILING
Measure training time, memory, FLOPs, and inference latency for:
1. Full fine-tuning (baseline)
2. LoRA r=8
3. 4-bit + LoRA r=8

Single task (MRPC), 3 runs for statistical reliability
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


def count_parameters(model):
    """Count trainable and total parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_latency(model, tokenized_dataset, device, num_samples=100):
    """Measure average inference latency using pre-tokenized dataset"""
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(tokenized_dataset))):
            sample = tokenized_dataset[i]
            
            # Create inputs from tokenized data
            inputs = {
                'input_ids': sample['input_ids'].unsqueeze(0).to(device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(device),
            }
            
            # Measure time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            _ = model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }


def estimate_flops(model, seq_length=128):
    """Estimate FLOPs for forward pass"""
    total_params, trainable_params = count_parameters(model)
    
    # Rough estimation for BERT-like models
    # Forward pass FLOPs ≈ 2 × params × seq_length
    forward_flops = 2 * total_params * seq_length
    
    # Training FLOPs ≈ 3 × forward (forward + backward + optimizer)
    training_flops = 3 * forward_flops
    
    return {
        "forward_gflops": forward_flops / 1e9,
        "training_gflops": training_flops / 1e9,
        "params_total": total_params,
        "params_trainable": trainable_params,
    }


def profile_method(method_name, seed=42):
    """Profile a single method"""
    logger.info(f"\n{'='*80}")
    logger.info(f"PROFILING: {method_name.upper()}, Seed: {seed}")
    logger.info(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model based on method
    if method_name == "full":
        logger.info("  Creating full fine-tuning model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        method_desc = "Full Fine-tuning"
        
    elif method_name == "lora":
        logger.info("  Creating LoRA model...")
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
        method_desc = "LoRA r=8"
        
    elif method_name == "4bit":
        logger.info("  Creating 4-bit quantized LoRA model...")
        try:
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
            method_desc = "4-bit + LoRA r=8"
        except Exception as e:
            logger.error(f"  ⚠️  bitsandbytes not installed or incompatible. Install with: pip install bitsandbytes")
            raise
    
    device = next(model.parameters()).device
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Estimate FLOPs
    flops = estimate_flops(model)
    logger.info(f"  Forward pass: {flops['forward_gflops']:.2f} GFLOPs")
    logger.info(f"  Training step: {flops['training_gflops']:.2f} GFLOPs")
    
    # Load dataset (MRPC for consistency)
    logger.info("  Loading MRPC dataset...")
    train_dataset = load_dataset("glue", "mrpc", split="train[:1000]")  # Subset for speed
    val_dataset = load_dataset("glue", "mrpc", split="validation[:100]")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize(batch):
        texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
    
    # Rename label column first
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    
    # Get columns to remove (everything except labels)
    train_cols = [col for col in train_dataset.column_names if col != "labels"]
    val_cols = [col for col in val_dataset.column_names if col != "labels"]
    
    # Tokenize and remove original columns
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_cols)
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=val_cols)
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    # Training profiling
    training_args = TrainingArguments(
        output_dir=f"checkpoints/profiling/{method_name}_{seed}",
        num_train_epochs=1,  # Just 1 epoch for profiling
        per_device_train_batch_size=16,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        fp16=(method_name != "4bit" and torch.cuda.is_available()),
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.01,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Measure training time
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    logger.info("  Training (1 epoch for profiling)...")
    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_memory = 0.0
    
    logger.info(f"  Training time: {train_time:.2f}s for 1 epoch on 1000 samples")
    logger.info(f"  Peak memory: {peak_memory:.2f} GB")
    
    # Extrapolate to full training
    samples_per_sec = len(train_dataset) / train_time
    estimated_full_time = (3668 * 3) / samples_per_sec  # MRPC full dataset, 3 epochs
    
    logger.info(f"  Samples/sec: {samples_per_sec:.2f}")
    logger.info(f"  Estimated full training: {estimated_full_time/60:.2f} minutes")
    
    # Measure inference latency (pass tokenized dataset)
    logger.info("  Measuring inference latency...")
    latency = measure_inference_latency(model, val_dataset, device)
    
    logger.info(f"  Inference latency (mean): {latency['mean_ms']:.2f} ms")
    logger.info(f"  Inference latency (p95): {latency['p95_ms']:.2f} ms")
    
    # Throughput (samples per second during inference)
    throughput = 1000 / latency['mean_ms']  # samples/sec
    logger.info(f"  Inference throughput: {throughput:.2f} samples/sec")
    
    result = {
        "method": method_name,
        "method_description": method_desc,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "trainable_percent": round(100 * trainable_params / total_params, 2),
        },
        "flops": {
            "forward_gflops": round(flops['forward_gflops'], 2),
            "training_gflops": round(flops['training_gflops'], 2),
        },
        "training": {
            "time_seconds_per_epoch": round(train_time, 2),
            "samples_per_second": round(samples_per_sec, 2),
            "estimated_full_training_minutes": round(estimated_full_time / 60, 2),
            "peak_memory_gb": round(peak_memory, 2),
            "training_loss": round(train_result.training_loss, 4),
        },
        "inference": {
            "latency_mean_ms": round(latency['mean_ms'], 2),
            "latency_std_ms": round(latency['std_ms'], 2),
            "latency_p50_ms": round(latency['p50_ms'], 2),
            "latency_p95_ms": round(latency['p95_ms'], 2),
            "latency_p99_ms": round(latency['p99_ms'], 2),
            "throughput_samples_per_sec": round(throughput, 2),
        },
    }
    
    logger.info(f"  ✅ {method_name.upper()} profiling complete!")
    
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def main():
    logger.info("=" * 100)
    logger.info("COMPUTATIONAL COST PROFILING")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("Methods: Full fine-tuning, LoRA r=8, 4-bit + LoRA")
    logger.info("Metrics: Training time, memory, FLOPs, inference latency")
    logger.info("=" * 100)
    
    results_dir = Path("results/profiling")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ["full", "lora", "4bit"]
    seeds = [42, 43, 44]  # 3 runs for statistical reliability
    
    all_results = {}
    start_time = time.time()
    
    for method in methods:
        for seed in seeds:
            try:
                result = profile_method(method, seed)
                all_results[f"{method}_seed{seed}"] = result
                
                # Save individual result
                with open(results_dir / f"profile_{method}_seed{seed}.json", 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                logger.error(f"❌ Error profiling {method} seed {seed}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Compute summary statistics
    summary = {}
    for method in methods:
        method_results = [v for k, v in all_results.items() if v['method'] == method]
        
        if method_results:
            summary[method] = {
                "training_time_mean": np.mean([r['training']['time_seconds_per_epoch'] for r in method_results]),
                "training_time_std": np.std([r['training']['time_seconds_per_epoch'] for r in method_results]),
                "memory_mean_gb": np.mean([r['training']['peak_memory_gb'] for r in method_results]),
                "memory_std_gb": np.std([r['training']['peak_memory_gb'] for r in method_results]),
                "inference_latency_mean_ms": np.mean([r['inference']['latency_mean_ms'] for r in method_results]),
                "inference_latency_std_ms": np.std([r['inference']['latency_mean_ms'] for r in method_results]),
                "throughput_mean": np.mean([r['inference']['throughput_samples_per_sec'] for r in method_results]),
                "params_trainable": method_results[0]['parameters']['trainable'],
                "forward_gflops": method_results[0]['flops']['forward_gflops'],
            }
    
    total_time = (time.time() - start_time) / 3600
    
    final_summary = {
        "total_experiments": len(all_results),
        "total_time_hours": round(total_time, 2),
        "summary_statistics": summary,
        "detailed_results": all_results,
    }
    
    with open(results_dir / "profiling_summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    logger.info("\n" + "=" * 100)
    logger.info("PROFILING COMPLETE!")
    logger.info("=" * 100)
    logger.info(f"Total time: {total_time:.2f}h")
    logger.info("\nSUMMARY:")
    for method, stats in summary.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Training: {stats['training_time_mean']:.2f}s ± {stats['training_time_std']:.2f}s")
        logger.info(f"  Memory: {stats['memory_mean_gb']:.2f} GB ± {stats['memory_std_gb']:.2f} GB")
        logger.info(f"  Inference: {stats['inference_latency_mean_ms']:.2f} ms ± {stats['inference_latency_std_ms']:.2f} ms")
        logger.info(f"  Throughput: {stats['throughput_mean']:.2f} samples/sec")
        logger.info(f"  Trainable params: {stats['params_trainable']:,}")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()