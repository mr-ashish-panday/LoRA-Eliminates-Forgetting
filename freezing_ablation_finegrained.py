"""
Fine-Grained Freezing Ablation: Characterize the Phase Transition
=================================================================
The original freezing ablation showed a dramatic cliff:
  50% frozen → 26.5% forgetting
  75% frozen → 22.5% forgetting
  95% frozen → 20.6% forgetting
  99.7% (LoRA) → 0.9% forgetting

The 95%→99.7% drop (19.7pp) is the most interesting unexplored finding.
This script fills in the gap with finer granularity.

BERT-base has 12 encoder layers + embeddings + classifier.
We control freezing at the layer level:

Layer mapping:
  - Embeddings (~23.8M params, always frozen in ablation)
  - Encoder layers 0-11 (~7.1M each, 85.1M total)
  - Classifier head (~1.5K params, always trainable)
  
Freeze configurations (BERT-specific):
  - 92% frozen: freeze layers 0-10, train layer 11 + classifier (~7.1M trainable)
  - 96% frozen: freeze 0-10 + half of layer 11 (~3.5M trainable)
  - Freeze 0-11, train ONLY classifier → ~99.99% frozen
  - We also test LoRA at different layer scopes

Fine-grained configs:
  1. freeze_11_of_12 = freeze 0-10, train layer 11 (=original 95% ≈ 92% actual)
  2. freeze_11_partial = freeze 0-10 + attention of 11, train FFN of 11
  3. freeze_all_12 = freeze 0-11, train ONLY classifier head
  4. lora_last_2 = freeze all, LoRA on layers 10-11 only
  5. lora_last_1 = freeze all, LoRA on layer 11 only

Seeds: 42, 43, 44
Estimated time: 3-5 hours
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time
import gc
import traceback

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/freezing_finegrained.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ========== FREEZING CONFIGURATIONS ==========

FREEZE_CONFIGS = {
    "freeze_10_of_12": {
        "description": "Freeze layers 0-9, train layers 10-11 + classifier (~85% frozen)",
        "freeze_layers": list(range(10)),
        "freeze_embeddings": True,
        "use_lora": False,
    },
    "freeze_11_of_12": {
        "description": "Freeze layers 0-10, train layer 11 + classifier (~92% frozen)",
        "freeze_layers": list(range(11)),
        "freeze_embeddings": True,
        "use_lora": False,
    },
    "freeze_11_partial_attn": {
        "description": "Freeze 0-10 + attention of layer 11, train only FFN of 11 + classifier (~96% frozen)",
        "freeze_layers": list(range(11)),
        "freeze_embeddings": True,
        "freeze_layer11_attention": True,
        "use_lora": False,
    },
    "freeze_all_12": {
        "description": "Freeze ALL 12 layers, train ONLY classifier head (~99.99% frozen)",
        "freeze_layers": list(range(12)),
        "freeze_embeddings": True,
        "use_lora": False,
    },
    "lora_layer_11_only": {
        "description": "LoRA applied ONLY to layer 11 (query+value), ~99.9% frozen",
        "freeze_layers": list(range(12)),
        "freeze_embeddings": True,
        "use_lora": True,
        "lora_layers": [11],
    },
    "lora_layers_10_11": {
        "description": "LoRA applied to layers 10-11 (query+value), ~99.8% frozen",
        "freeze_layers": list(range(12)),
        "freeze_embeddings": True,
        "use_lora": True,
        "lora_layers": [10, 11],
    },
    "lora_all_layers": {
        "description": "LoRA applied to ALL layers (query+value), ~99.7% frozen (standard LoRA)",
        "freeze_layers": list(range(12)),
        "freeze_embeddings": True,
        "use_lora": True,
        "lora_layers": list(range(12)),
    },
}


def apply_freeze_config(model, config_name, config, seed):
    """Apply a specific freezing configuration to the model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if config.get("use_lora"):
        # For LoRA configs, first freeze everything then add LoRA
        for param in model.parameters():
            param.requires_grad = False
        
        # Build target module names for specific layers
        lora_layers = config.get("lora_layers", list(range(12)))
        target_modules = []
        for layer_idx in lora_layers:
            target_modules.append(f"bert.encoder.layer.{layer_idx}.attention.self.query")
            target_modules.append(f"bert.encoder.layer.{layer_idx}.attention.self.value")
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        
    else:
        # Standard layer freezing
        
        # Freeze embeddings if specified
        if config.get("freeze_embeddings"):
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze specified layers
        for layer_idx in config.get("freeze_layers", []):
            for param in model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        
        # Special: freeze attention of layer 11 only
        if config.get("freeze_layer11_attention"):
            for param in model.bert.encoder.layer[11].attention.parameters():
                param.requires_grad = False
        
        # Classifier always trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen_pct = 100 * (1 - trainable / total)
    
    logger.info(f"  Config: {config_name}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"  Frozen: {frozen_pct:.2f}%")
    
    return model, trainable, total, frozen_pct


# ========== DATA AND EVAL (same as phase7) ==========

def load_and_prepare_dataset(task, tokenizer, split="train", max_samples=None):
    if split == "train":
        dataset = load_dataset("glue", task, split="train")
        if max_samples and len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
    else:
        dataset = load_dataset("glue", task, split="validation")
    
    def tokenize(batch):
        if "sentence2" in batch:
            texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
        else:
            texts = batch["sentence"]
        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
    
    dataset = dataset.rename_column("label", "labels")
    cols = [col for col in dataset.column_names if col != "labels"]
    dataset = dataset.map(tokenize, batched=True, remove_columns=cols)
    dataset.set_format("torch")
    return dataset


def evaluate_on_task(model, dataset, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**sample)
            if outputs.logits.argmax(dim=1).item() == sample["labels"].item():
                correct += 1
    return correct / len(dataset)


# ========== SEQUENTIAL TRAINING ==========

def run_sequential_training(config_name, config, seed, tasks, tokenizer):
    """Run sequential training with a specific freeze config"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Config: {config_name}, Seed: {seed}")
    logger.info(f"{'='*80}")
    
    # Create base model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Apply freezing
    model, trainable, total, frozen_pct = apply_freeze_config(model, config_name, config, seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load validation sets
    val_datasets = {}
    for task in tasks:
        val_datasets[task] = load_and_prepare_dataset(task, tokenizer, split="val")
    
    accuracy_matrix = {task: {} for task in tasks}
    
    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")
        
        train_dataset = load_and_prepare_dataset(
            current_task, tokenizer, split="train",
            max_samples=2000 if current_task == "sst2" else None
        )
        
        training_args = TrainingArguments(
            output_dir=f"checkpoints/freezing_fg/{config_name}_{seed}_{current_task}",
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
        
        # Evaluate
        for eval_task in tasks[:i+1]:
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            logger.info(f"  {eval_task}: {acc:.4f}")
        
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
        "avg_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "max_forgetting": float(np.max(forgetting)) if forgetting else 0.0,
        "forgetting_per_task": {tasks[i]: float(forgetting[i]) for i in range(len(forgetting))},
        "trainable_params": trainable,
        "total_params": total,
        "frozen_percentage": frozen_pct,
    }


# ========== MAIN ==========

def run_finegrained_ablation():
    logger.info("=" * 100)
    logger.info("FINE-GRAINED FREEZING ABLATION: Phase Transition Characterization")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    
    results_dir = Path("results/freezing_ablation_finegrained")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    seeds = [42, 43, 44]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    all_results = {}
    start_time = time.time()
    experiment_count = 0
    total_experiments = len(FREEZE_CONFIGS) * len(seeds)
    
    for config_name, config in FREEZE_CONFIGS.items():
        for seed in seeds:
            experiment_count += 1
            logger.info(f"\n{'#'*100}")
            logger.info(f"EXPERIMENT {experiment_count}/{total_experiments}: {config_name}, Seed {seed}")
            logger.info(f"{'#'*100}")
            
            try:
                exp_start = time.time()
                result = run_sequential_training(config_name, config, seed, tasks, tokenizer)
                exp_time = (time.time() - exp_start) / 3600
                
                key = f"{config_name}_seed{seed}"
                all_results[key] = {
                    "config_name": config_name,
                    "description": config["description"],
                    "seed": seed,
                    "frozen_percentage": round(result["frozen_percentage"], 2),
                    "trainable_params": result["trainable_params"],
                    "total_params": result["total_params"],
                    "accuracy_matrix": result["accuracy_matrix"],
                    "avg_forgetting": round(result["avg_forgetting"], 4),
                    "max_forgetting": round(result["max_forgetting"], 4),
                    "forgetting_per_task": {k: round(v, 4) for k, v in result["forgetting_per_task"].items()},
                    "time_hours": round(exp_time, 4),
                }
                
                with open(results_dir / f"{key}.json", 'w') as f:
                    json.dump(all_results[key], f, indent=2)
                
                logger.info(f"\n✅ Complete: Frozen={result['frozen_percentage']:.1f}%, Forgetting={result['avg_forgetting']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                logger.error(traceback.format_exc())
            
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    total_time = (time.time() - start_time) / 3600
    
    # Create summary with phase transition analysis
    configs_summary = {}
    for config_name in FREEZE_CONFIGS:
        config_results = [v for k, v in all_results.items() if config_name in k]
        if config_results:
            forgetting_vals = [r["avg_forgetting"] for r in config_results]
            frozen_pcts = [r["frozen_percentage"] for r in config_results]
            configs_summary[config_name] = {
                "description": FREEZE_CONFIGS[config_name]["description"],
                "frozen_percentage": round(np.mean(frozen_pcts), 2),
                "avg_forgetting_mean": round(np.mean(forgetting_vals), 4),
                "avg_forgetting_std": round(np.std(forgetting_vals), 4),
                "n_seeds": len(config_results),
            }
    
    summary = {
        "experiment": "Fine-Grained Freezing Ablation",
        "total_time_hours": round(total_time, 2),
        "total_experiments": total_experiments,
        "successful": len(all_results),
        "phase_transition_summary": configs_summary,
        "results": all_results,
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print phase transition table
    print("\n\n" + "="*80)
    print("PHASE TRANSITION SUMMARY")
    print("="*80)
    print(f"{'Config':<30} {'Frozen%':>8} {'Forgetting':>12} {'Std':>8}")
    print("-"*58)
    for name, data in sorted(configs_summary.items(), key=lambda x: x[1]["frozen_percentage"]):
        print(f"{name:<30} {data['frozen_percentage']:>7.1f}% {data['avg_forgetting_mean']:>11.4f} {data['avg_forgetting_std']:>7.4f}")
    
    logger.info(f"\n{'='*100}")
    logger.info(f"FINE-GRAINED ABLATION COMPLETED! Time: {total_time:.2f}h")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_finegrained_ablation()
