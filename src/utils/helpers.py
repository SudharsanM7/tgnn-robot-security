"""
Helper functions and utilities
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class ConfigLoader:
    """Load and manage configuration files."""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            config_path: Path to YAML config
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def save_yaml(config: Dict, output_path: str):
        """Save configuration to YAML file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer,
                       epoch: int, metrics: Dict, best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Metrics dictionary
            best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Regular checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        
        # Best checkpoint
        if best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       best: bool = True):
        """
        Load model checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load state into
            best: Load best or latest checkpoint
            
        Returns:
            Loaded checkpoint dictionary
        """
        if best:
            ckpt_path = self.checkpoint_dir / "best_model.pt"
        else:
            # Find latest checkpoint
            ckpts = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not ckpts:
                raise FileNotFoundError("No checkpoints found")
            ckpt_path = max(ckpts, key=lambda p: int(p.stem.split('_')[-1]))
        
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        return checkpoint


class AverageMeter:
    """Accumulate and compute averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
