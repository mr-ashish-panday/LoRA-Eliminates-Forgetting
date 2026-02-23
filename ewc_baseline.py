"""
EWC Baseline: Elastic Weight Consolidation Under Our Protocol
=============================================================
Implements EWC (Kirkpatrick et al., 2017) under the EXACT same conditions
as our LoRA experiments, making Table 8 comparison fair.

Same protocol:
  - Task sequence: RTE → MRPC → CoLA → SST2
  - Training: 3 epochs, lr=2e-5, batch=16, warmup=100
  - Seeds: 42, 43, 44
  - Evaluation: Full validation sets on all previous tasks

EWC specifics:
  - Fisher information computed on 500 training samples per task
  - Lambda (EWC strength) grid: [100, 1000, 5000]
  - After each task, compute Fisher information and store optimal weights

3 lambdas × 3 seeds = 9 experiments
Estimated time: 4-6 hours
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time
import gc
import traceback
import copy

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
)
from datasets import load_dataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ewc_baseline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ========== EWC IMPLEMENTATION ==========

class EWC:
    """
    Elastic Weight Consolidation
    
    After training on a task, compute Fisher information matrix (diagonal approx)
    and store the optimal parameters. For subsequent tasks, add a penalty term 
    that prevents parameters from moving too far from stored values.
    """
    
    def __init__(self, model, ewc_lambda=1000):
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Store Fisher information and optimal params per task
        self.fisher_info = {}  # task_name -> {param_name: fisher_diagonal}
        self.optimal_params = {}  # task_name -> {param_name: param_values}
        self.tasks_seen = []
    
    def compute_fisher(self, task_name, dataloader, device, n_samples=500):
        """
        Compute diagonal Fisher Information Matrix using empirical Fisher.
        Uses gradients of the log-likelihood on the training data.
        """
        logger.info(f"  Computing Fisher information for task '{task_name}' ({n_samples} samples)...")
        
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= n_samples:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            self.model.zero_grad()
            
            outputs = self.model(**batch)
            # Use log-softmax of predicted class (empirical Fisher)
            log_probs = F.log_softmax(outputs.logits, dim=1)
            labels = batch["labels"]
            loss = F.nll_loss(log_probs, labels)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            
            sample_count += batch["labels"].size(0)
        
        # Normalize
        for n in fisher:
            fisher[n] /= sample_count
        
        # Store
        self.fisher_info[task_name] = fisher
        self.optimal_params[task_name] = {
            n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.tasks_seen.append(task_name)
        
        logger.info(f"  Fisher computation complete. Tasks seen: {self.tasks_seen}")
    
    def penalty(self):
        """
        Compute EWC penalty: sum over all previous tasks of 
        lambda/2 * Fisher * (params - optimal_params)^2
        """
        total_penalty = 0.0
        
        for task_name in self.tasks_seen:
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in self.fisher_info[task_name]:
                    penalty = self.fisher_info[task_name][n] * (p - self.optimal_params[task_name][n]) ** 2
                    total_penalty += penalty.sum()
        
        return (self.ewc_lambda / 2) * total_penalty


# ========== DATA LOADING ==========

def load_and_prepare_dataset(task, tokenizer, split="train", max_samples=None):
    """Load and tokenize dataset"""
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
    """Evaluate model on a task dataset"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()}
            outputs = model(**sample)
            if outputs.logits.argmax(dim=1).item() == sample["labels"].item():
                correct += 1
    return correct / len(dataset)


# ========== SEQUENTIAL TRAINING WITH EWC ==========

def run_ewc_sequential(ewc_lambda, seed, tasks, tokenizer):
    """
    Train sequentially with EWC regularization.
    Uses manual training loop (not HF Trainer) to inject EWC penalty.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"EWC Training: Lambda={ewc_lambda}, Seed={seed}")
    logger.info(f"Task Order: {' → '.join(tasks)}")
    logger.info(f"{'='*80}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize EWC
    ewc = EWC(model, ewc_lambda=ewc_lambda)
    
    # Load validation sets
    val_datasets = {}
    for task in tasks:
        val_datasets[task] = load_and_prepare_dataset(task, tokenizer, split="val")
    
    accuracy_matrix = {task: {} for task in tasks}
    
    for i, current_task in enumerate(tasks):
        logger.info(f"\n--- Training Task {i+1}/{len(tasks)}: {current_task} ---")
        
        # Load training data
        train_dataset = load_and_prepare_dataset(
            current_task, tokenizer, split="train",
            max_samples=2000 if current_task == "sst2" else None
        )
        logger.info(f"  Training samples: {len(train_dataset)}")
        
        # Create DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2e-5,
            weight_decay=0.01
        )
        
        # Training loop
        model.train()
        num_epochs = 3
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = 100
        
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_ewc_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                task_loss = outputs.loss
                
                # EWC penalty (only if we've seen previous tasks)
                ewc_loss = ewc.penalty() if ewc.tasks_seen else torch.tensor(0.0)
                
                # Total loss
                total_loss = task_loss + ewc_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += task_loss.item()
                epoch_ewc_loss += ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss
                num_batches += 1
                global_step += 1
            
            avg_loss = epoch_loss / num_batches
            avg_ewc = epoch_ewc_loss / num_batches
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: task_loss={avg_loss:.4f}, ewc_loss={avg_ewc:.4f}")
        
        # Compute Fisher information for this task (BEFORE evaluating)
        fisher_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        ewc.compute_fisher(current_task, fisher_dataloader, device, n_samples=500)
        
        # Evaluate on ALL previous tasks
        logger.info(f"  Evaluating on all tasks seen so far:")
        for j, eval_task in enumerate(tasks[:i+1]):
            acc = evaluate_on_task(model, val_datasets[eval_task], device)
            accuracy_matrix[eval_task][f"after_{current_task}"] = acc
            
            if j < i:
                initial_acc = accuracy_matrix[eval_task][f"after_{eval_task}"]
                forgetting = initial_acc - acc
                logger.info(f"    {eval_task}: {acc:.4f} (Initial: {initial_acc:.4f}, Forgot: {forgetting:.4f})")
            else:
                logger.info(f"    {eval_task}: {acc:.4f} (just trained)")
        
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
        "final_avg_accuracy": float(np.mean([accuracy_matrix[t][f"after_{tasks[-1]}"] for t in tasks])),
        "initial_avg_accuracy": float(np.mean([accuracy_matrix[t][f"after_{t}"] for t in tasks])),
    }


# ========== MAIN ==========

def run_ewc_baseline():
    logger.info("=" * 100)
    logger.info("EWC BASELINE: Elastic Weight Consolidation Under Our Protocol")
    logger.info("=" * 100)
    logger.info(f"Start: {datetime.now().isoformat()}")
    
    results_dir = Path("results/ewc_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    tasks = ["rte", "mrpc", "cola", "sst2"]
    lambdas = [100, 1000, 5000]
    seeds = [42, 43, 44]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    all_results = {}
    total = len(lambdas) * len(seeds)
    completed = 0
    start_time = time.time()
    
    for ewc_lambda in lambdas:
        for seed in seeds:
            completed += 1
            logger.info(f"\n{'#'*100}")
            logger.info(f"EXPERIMENT {completed}/{total}: Lambda={ewc_lambda}, Seed={seed}")
            logger.info(f"{'#'*100}")
            
            try:
                exp_start = time.time()
                
                result = run_ewc_sequential(ewc_lambda, seed, tasks, tokenizer)
                exp_time = (time.time() - exp_start) / 3600
                
                key = f"ewc_lambda{ewc_lambda}_seed{seed}"
                all_results[key] = {
                    "method": "EWC",
                    "ewc_lambda": ewc_lambda,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "task_sequence": tasks,
                    "accuracy_matrix": result["accuracy_matrix"],
                    "avg_forgetting": round(result["avg_forgetting"], 4),
                    "max_forgetting": round(result["max_forgetting"], 4),
                    "forgetting_per_task": {k: round(v, 4) for k, v in result["forgetting_per_task"].items()},
                    "final_avg_accuracy": round(result["final_avg_accuracy"], 4),
                    "initial_avg_accuracy": round(result["initial_avg_accuracy"], 4),
                    "time_hours": round(exp_time, 4),
                }
                
                with open(results_dir / f"{key}.json", 'w') as f:
                    json.dump(all_results[key], f, indent=2)
                
                logger.info(f"\n✅ Complete: Forgetting={result['avg_forgetting']:.4f}, Acc={result['final_avg_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                logger.error(traceback.format_exc())
            
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    total_time = (time.time() - start_time) / 3600
    
    # Summary by lambda
    lambda_summary = {}
    for lam in lambdas:
        lam_results = [v for k, v in all_results.items() if f"lambda{lam}" in k]
        if lam_results:
            forgetting_vals = [r["avg_forgetting"] for r in lam_results]
            accuracy_vals = [r["final_avg_accuracy"] for r in lam_results]
            lambda_summary[f"lambda_{lam}"] = {
                "avg_forgetting_mean": round(np.mean(forgetting_vals), 4),
                "avg_forgetting_std": round(np.std(forgetting_vals), 4),
                "final_accuracy_mean": round(np.mean(accuracy_vals), 4),
                "final_accuracy_std": round(np.std(accuracy_vals), 4),
                "n_seeds": len(lam_results),
            }
    
    summary = {
        "experiment": "EWC Baseline",
        "total_time_hours": round(total_time, 2),
        "total_experiments": total,
        "successful": len(all_results),
        "lambda_summary": lambda_summary,
        "results": all_results,
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table
    print("\n\n" + "="*70)
    print("EWC RESULTS vs PAPER METHODS")
    print("="*70)
    print(f"{'Method':<25} {'Forgetting':>12} {'Std':>8} {'Accuracy':>10}")
    print("-"*55)
    for lam_name, data in lambda_summary.items():
        print(f"EWC ({lam_name}){'':<10} {data['avg_forgetting_mean']:>11.4f} {data['avg_forgetting_std']:>7.4f} {data['final_accuracy_mean']:>9.4f}")
    print("-"*55)
    print(f"{'Full FT (paper)':<25} {'0.2083':>12} {'0.033':>8} {'0.810':>10}")
    print(f"{'LoRA (paper)':<25} {'0.0094':>12} {'0.027':>8} {'0.684':>10}")
    print(f"{'4-bit+LoRA (paper)':<25} {'0.0144':>12} {'0.010':>8} {'0.715':>10}")
    
    logger.info(f"\n{'='*100}")
    logger.info(f"EWC BASELINE COMPLETED! Time: {total_time:.2f}h")
    logger.info(f"{'='*100}")


if __name__ == "__main__":
    run_ewc_baseline()
