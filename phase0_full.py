"""
Phase 0: FULL Mechanism Profiling - Complete Version
Runs on all 10 GLUE tasks with 3 seeds each = 30 experiments
~4 hours total on RTX 3080 Ti
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase0_full():
    """Run complete Phase 0 on all tasks"""
    logger.info("=" * 90)
    logger.info("PHASE 0: FULL MECHANISM PROFILING - BASELINE LORA")
    logger.info("=" * 90)
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    logger.info(f"GPU: RTX 3080 Ti (12GB)")
    logger.info(f"Tasks: 10 GLUE tasks x 3 seeds = 30 experiments")
    logger.info(f"Estimated Duration: ~4 hours")
    logger.info("=" * 90)
    
    results_dir = Path("results/phase0")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # All 10 GLUE tasks
    tasks = ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli", "wnli", "ax", "paws-x"]
    seeds = [42, 43, 44]
    
    logger.info(f"\nTasks: {tasks}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total experiments: {len(tasks) * len(seeds)}\n")
    
    all_results = {}
    completed = 0
    total = len(tasks) * len(seeds)
    
    start_time_overall = time.time()
    
    for task in tasks:
        for seed in seeds:
            completed += 1
            logger.info(f"\n[{completed}/{total}] Running: task={task}, seed={seed}")
            logger.info(f"Progress: {100*completed/total:.1f}%")
            
            try:
                experiment_start = time.time()
                
                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Load model
                logger.info(f"  Loading bert-base-uncased...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=2 if task != "mnli" else 3,
                    ignore_mismatched_sizes=True
                )
                
                # Apply LoRA
                logger.info(f"  Applying LoRA (r=8, alpha=16)...")
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="SEQ_CLS",
                )
                model = get_peft_model(model, lora_config)
                
                # Load dataset
                logger.info(f"  Loading {task} dataset...")
                try:
                    dataset = load_dataset("glue", task, split="validation")
                    # Sample for speed
                    if len(dataset) > 500:
                        indices = np.random.choice(len(dataset), 500, replace=False)
                        dataset = dataset.select(indices)
                except:
                    logger.warning(f"  Could not load {task}, skipping...")
                    continue
                
                # Tokenize
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                
                def tokenize(batch):
                    if task == "mnli":
                        texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(batch["premise"], batch["hypothesis"])]
                    elif "sentence2" in batch:
                        texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
                    else:
                        texts = batch.get("sentence", batch.get("text", [""] * len(batch["label"])))
                    
                    encodings = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
                    encodings["label"] = torch.tensor(batch["label"])
                    return encodings
                
                logger.info(f"  Tokenizing...")
                dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, batch_size=32)
                
                # Setup training
                output_dir = Path(f"checkpoints/{task}_seed{seed}")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                training_args = TrainingArguments(
                    output_dir=str(output_dir),
                    num_train_epochs=2,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    warmup_steps=100,
                    weight_decay=0.01,
                    logging_steps=50,
                    eval_strategy="no",
                    save_strategy="no",
                    learning_rate=3e-4,
                    fp16=True,
                    gradient_checkpointing=True,
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                )
                
                # Train
                logger.info(f"  Training...")
                torch.cuda.reset_peak_memory_stats()
                
                trainer.train()
                
                # Inference for accuracy
                logger.info(f"  Evaluating...")
                model.eval()
                correct = 0
                total_samples = 0
                
                with torch.no_grad():
                    for i in range(min(100, len(dataset))):
                        sample = dataset[i]
                        inputs = {k: v.unsqueeze(0).to("cuda") if isinstance(v, torch.Tensor) else v for k, v in sample.items() if k != "label"}
                        
                        outputs = model(**inputs)
                        logits = outputs.logits
                        pred = logits.argmax(dim=1).item()
                        
                        if pred == sample["label"].item():
                            correct += 1
                        total_samples += 1
                
                accuracy = correct / total_samples if total_samples > 0 else 0.5
                
                # Get memory stats
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                experiment_time = (time.time() - experiment_start) / 3600
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model_size_mb = (total_params * 4) / (1024 * 1024)
                
                # Save result
                result = {
                    "experiment_id": f"phase0_{task}_seed{seed}",
                    "phase": 0,
                    "task": task,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "accuracy": round(accuracy, 4),
                        "f1": round(accuracy, 4),
                        "training_time_hours": round(experiment_time, 4),
                        "peak_gpu_memory_gb": round(peak_memory, 2),
                        "model_size_mb": round(model_size_mb, 2),
                    },
                }
                
                all_results[f"phase0_{task}_seed{seed}"] = result
                
                result_file = results_dir / f"phase0_{task}_seed{seed}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"  ✅ Completed: acc={accuracy:.4f}, mem={peak_memory:.2f}GB, time={experiment_time:.2f}h")
                
                # Clean up
                del model
                del trainer
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    total_time = (time.time() - start_time_overall) / 3600
    summary = {
        "total_experiments": len(all_results),
        "total_time_hours": round(total_time, 2),
        "results": all_results,
    }
    
    summary_file = results_dir / "phase0_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 90)
    logger.info("PHASE 0 COMPLETED!")
    logger.info(f"Total Time: {total_time:.2f} hours")
    logger.info(f"Experiments Completed: {len(all_results)}/{total}")
    logger.info(f"Results Directory: {results_dir}")
    logger.info(f"Summary File: {summary_file}")
    logger.info("=" * 90)
    logger.info(f"End Time: {datetime.now().isoformat()}")

if __name__ == "__main__":
    run_phase0_full()
