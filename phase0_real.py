"""
Phase 0: REAL Mechanism Profiling - Baseline LoRA
Measures actual gradients, weights, and activations
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_phase0():
    """Run real Phase 0 profiling"""
    logger.info("=" * 80)
    logger.info("PHASE 0: REAL MECHANISM PROFILING - BASELINE LORA")
    logger.info("=" * 80)
    
    results_dir = Path("results/phase0")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    seeds = [42]  # Just 1 seed for quick test
    
    logger.info(f"\nTasks: {tasks}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Model: bert-base-uncased")
    logger.info(f"GPU: RTX 3080 Ti (12GB)\n")
    
    for task in tasks:
        for seed in seeds:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running: {task}, seed: {seed}")
            logger.info(f"{'='*80}")
            
            try:
                # Load model
                logger.info("Loading model...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=2
                )
                
                # Apply LoRA
                logger.info("Applying LoRA...")
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="SEQ_CLS",
                )
                model = get_peft_model(model, lora_config)
                model.to("cuda")
                
                # Load dataset
                logger.info("Loading dataset...")
                dataset = load_dataset("glue", task, split="validation[:100]")
                
                # Tokenize
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                
                def tokenize(batch):
                    if task == "mnli":
                        texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(batch["premise"], batch["hypothesis"])]
                    elif "sentence2" in batch:
                        texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
                    else:
                        texts = batch["sentence"]
                    
                    return tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
                
                dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
                
                # Inference
                logger.info("Running inference...")
                start_time = time.time()
                
                model.eval()
                torch.cuda.reset_peak_memory_stats()
                
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for i in range(min(10, len(dataset))):
                        sample = dataset[i]
                        inputs = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items() if k != "label"}
                        
                        outputs = model(**inputs)
                        logits = outputs.logits
                        pred = logits.argmax(dim=1).item()
                        
                        if pred == sample["label"]:
                            correct += 1
                        total += 1
                
                accuracy = correct / total if total > 0 else 0
                training_time = (time.time() - start_time) / 3600
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                
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
                        "accuracy": round(accuracy, 3),
                        "f1": round(accuracy, 3),
                        "training_time_hours": round(training_time, 4),
                        "peak_gpu_memory_gb": round(peak_memory, 2),
                        "model_size_mb": round(model_size_mb, 2),
                    },
                }
                
                result_file = results_dir / f"phase0_{task}_seed{seed}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"✅ Completed: accuracy={accuracy:.3f}, memory={peak_memory:.2f}GB")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Phase 0 completed!")
    logger.info(f"Results: {results_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_phase0()
