"""
Unified PEFT Framework - Main Configuration
Core settings for all experiments
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class ExperimentConfig:
    """Base experiment configuration"""
    
    # Project paths
    project_root: str = "/tmp/unified_peft_framework"
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    # Phase identifier
    phase: int = 0
    phase_name: str = "mechanism_profiling"
    
    # Model configuration
    model_name: str = "bert-base-uncased"
    model_size: str = "base"
    max_seq_length: int = 128
    
    # Task configuration
    tasks: List[str] = None
    num_tasks: int = 10
    
    # Training configuration
    num_epochs: int = 3
    per_device_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Random seed
    seed: int = 42
    num_seeds: int = 3
    
    # PEFT configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Quantization configuration
    quantization_enabled: bool = False
    quantization_bits: int = 8
    
    # Hierarchical ranks configuration
    hierarchical_enabled: bool = False
    hierarchical_strategy: str = "linear_decay"
    
    # Module selection configuration
    module_selection_enabled: bool = False
    selected_modules: List[str] = None
    
    # Extended components configuration
    extended_components_enabled: bool = False
    
    # Hardware
    device: str = "cuda"
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    
    # Reproducibility
    deterministic: bool = True
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = [
                "rte", "mrpc", "cola", "sst2", "qnli",
                "qqp", "mnli", "wnli", "ax", "paws-x"
            ][:self.num_tasks]
        
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
        
        if self.selected_modules is None:
            self.selected_modules = ["q_proj", "v_proj"]
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save config to JSON"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load config from JSON"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Default configurations for each phase
PHASE_0_CONFIG = ExperimentConfig(
    phase=0,
    phase_name="mechanism_profiling",
    num_epochs=3,
    num_seeds=3,
    per_device_batch_size=32,
)

PHASE_CONFIGS = {
    "phase_0": PHASE_0_CONFIG,
}
