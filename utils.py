"""
Unified PEFT Framework - Utilities
Helper functions for experiments
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=level,
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)


class ResultsManager:
    """Manage experiment results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def add_result(self, phase: str, task: str, seed: int, metrics: Dict[str, Any]):
        """Add experiment result"""
        key = f"{phase}_{task}_seed{seed}"
        self.results[key] = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "task": task,
            "seed": seed,
            "metrics": metrics,
        }
    
    def save_result(self, phase: str, task: str, seed: int, metrics: Dict[str, Any], 
                    filename: Optional[str] = None):
        """Save single result to JSON"""
        self.add_result(phase, task, seed, metrics)
        
        if filename is None:
            filename = f"{phase}_{task}_seed{seed}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results[f"{phase}_{task}_seed{seed}"], f, indent=2)
        
        return filepath
    
    def save_all(self, filename: str = "all_results.json"):
        """Save all results to single JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        return filepath
    
    def get_summary(self, phase: str, task: Optional[str] = None) -> Dict:
        """Get summary statistics for phase/task"""
        results = [v for k, v in self.results.items() if phase in k and (task is None or task in k)]
        
        if not results:
            return {}
        
        all_metrics = {}
        for result in results:
            for metric_name, metric_value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        summary = {}
        for metric_name, values in all_metrics.items():
            if isinstance(values[0], (int, float)):
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        return summary


class MetricsCalculator:
    """Calculate standard metrics"""
    
    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        """Compute standard metrics for tasks"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
        }


class HardwareMonitor:
    """Monitor GPU and CPU usage"""
    
    def __init__(self):
        self.measurements = []
    
    def record(self):
        """Record current hardware usage"""
        measurement = {
            'timestamp': datetime.now().isoformat(),
            'gpu_memory_allocated_gb': 0,
            'gpu_memory_reserved_gb': 0,
            'cpu_memory_percent': 0,
        }
        
        if torch.cuda.is_available():
            measurement['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            measurement['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
        
        try:
            import psutil
            measurement['cpu_memory_percent'] = psutil.virtual_memory().percent
        except ImportError:
            pass
        
        self.measurements.append(measurement)
        return measurement
    
    def get_peak_memory(self):
        """Get peak GPU memory usage"""
        if not self.measurements:
            return 0
        
        gpu_memories = [m['gpu_memory_allocated_gb'] for m in self.measurements]
        return max(gpu_memories) if gpu_memories else 0


def get_device_info() -> Dict[str, str]:
    """Get device information"""
    info = {
        'device': 'cpu',
        'gpu_name': 'N/A',
        'gpu_memory_gb': '0',
        'cuda_version': 'N/A',
        'pytorch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info['device'] = f'cuda:{torch.cuda.current_device()}'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = str(torch.cuda.get_device_properties(0).total_memory / 1e9)
        info['cuda_version'] = torch.version.cuda
    
    return info
