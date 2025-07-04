import argparse
import yaml
import wandb
import logging
import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from curriculum import CurriculumTrainer
from torch.utils.tensorboard import SummaryWriter

def setup_logging(log_dir, experiment_name, log_level=logging.INFO):
    """Setup comprehensive logging system"""
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)15s | %(funcName)20s:%(lineno)4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for detailed logs
    detailed_log_file = log_dir / f"{experiment_name}_detailed.log"
    file_handler = logging.FileHandler(detailed_log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error log file
    error_log_file = log_dir / f"{experiment_name}_errors.log"
    error_handler = logging.FileHandler(error_log_file, mode='w')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Training metrics log (JSON format)
    metrics_log_file = log_dir / f"{experiment_name}_metrics.jsonl"
    
    logging.info(f"Logging initialized:")
    logging.info(f"  - Detailed log: {detailed_log_file}")
    logging.info(f"  - Error log: {error_log_file}")
    logging.info(f"  - Metrics log: {metrics_log_file}")
    
    return metrics_log_file

def setup_tensorboard(log_dir, experiment_name):
    """Setup TensorBoard logging"""
    tb_dir = Path(log_dir) / "tensorboard" / experiment_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    logging.info(f"TensorBoard logs: {tb_dir}")
    return writer

def log_system_info():
    """Log system and environment information"""
    logging.info("=== SYSTEM INFORMATION ===")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"  GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
            logging.info(f"    Compute capability: {props.major}.{props.minor}")
    
    # Log memory usage
    import psutil
    memory = psutil.virtual_memory()
    logging.info(f"System RAM: {memory.total // 1024**3} GB (Available: {memory.available // 1024**3} GB)")
    logging.info("=" * 50)

def log_model_info(model):
    """Log detailed model information"""
    logging.info("=== MODEL INFORMATION ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Log parameter breakdown by component
    component_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_params[name] = params
        logging.info(f"  {name}: {params:,} parameters")
    
    # Log model size estimation
    param_size = total_params * 4  # 4 bytes per float32 parameter
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / 1024**2  # Convert to MB
    
    logging.info(f"Estimated model size: {total_size:.2f} MB")
    logging.info("=" * 50)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'component_params': component_params,
        'model_size_mb': total_size
    }

def log_config_info(cfg, config_dict):
    """Log configuration information"""
    logging.info("=== CONFIGURATION ===")
    
    def log_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                logging.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logging.info(f"{prefix}{key}: {value}")
    
    log_dict(config_dict)
    logging.info("=" * 50)

def save_checkpoint(model, optimizer, epoch, phase, loss, checkpoint_dir, experiment_name, is_best=False):
    """Save model checkpoint with comprehensive metadata"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"{experiment_name}_phase{phase}_epoch{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / f"{experiment_name}_best.pt"
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint: {best_path}")
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / f"{experiment_name}_latest.pt"
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def log_training_metrics(metrics, metrics_log_file, tb_writer, global_step):
    """Log training metrics to multiple destinations"""
    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['global_step'] = global_step
    
    # Log to file (JSONL format)
    with open(metrics_log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    # Log to TensorBoard
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tb_writer.add_scalar(key, value, global_step)
    
    # Log to console (key metrics only)
    if 'phase' in metrics and 'epoch' in metrics:
        key_metrics = {k: v for k, v in metrics.items() 
                      if k in ['phase', 'epoch', 'loss', 'phase1_loss', 'phase2_loss', 'phase3_loss', 'phase4_loss']}
        logging.info(f"Metrics: {key_metrics}")

def monitor_gpu_memory():
    """Monitor and log GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            cached = torch.cuda.memory_reserved(i) / 1024**2  # MB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**2  # MB
            
            logging.debug(f"GPU {i} Memory - Allocated: {allocated:.1f}MB, "
                         f"Cached: {cached:.1f}MB, Max: {max_allocated:.1f}MB")
            
            return {
                f'gpu_{i}_memory_allocated_mb': allocated,
                f'gpu_{i}_memory_cached_mb': cached,
                f'gpu_{i}_memory_max_mb': max_allocated
            }
    return {}

def load_config(config_path):
    """Load YAML configuration file with validation"""
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise

class Config:
    """Simple config class to replace OmegaConf with better logging"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

class EnhancedCurriculumTrainer(CurriculumTrainer):
    """Enhanced trainer with extensive logging capabilities"""
    
    def __init__(self, cfg, metrics_log_file, tb_writer, checkpoint_dir, experiment_name):
        super().__init__(cfg)
        self.metrics_log_file = metrics_log_file
        self.tb_writer = tb_writer
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.global_step = 0
        self.best_loss = float('inf')
        self.phase_start_time = None
        
        # Log model information
        self.model_info = log_model_info(self.model)
        
    def train_all(self):
        """Enhanced training with comprehensive logging"""
        logging.info("=== STARTING TRAINING ===")
        training_start_time = time.time()
        
        phases = [
            ("Phase 1: Unimodal", self.phase1_unimodal),
            ("Phase 2: Alignment", self.phase2_alignment),
            ("Phase 3: Mask Warmup", self.phase3_mask_warmup),
            ("Phase 4: Full Training", self.phase4_full)
        ]
        
        # Get training summary from parent class
        training_summary = super().train_all()
        
        # Log final training metrics
        total_training_time = time.time() - training_start_time
        logging.info(f"\n{'='*60}")
        logging.info(f"TRAINING COMPLETED!")
        logging.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/3600:.2f} hours)")
        logging.info(f"Best loss achieved: {self.best_loss:.6f}")
        logging.info(f"{'='*60}")
        
        # Final model save
        save_checkpoint(
            self.model, self.opt, "final", "all", self.best_loss,
            self.checkpoint_dir, self.experiment_name, is_best=True
        )
        
        # Ensure training_summary is not None
        if training_summary is None:
            training_summary = {}
            
        return {
            'total_training_time': total_training_time,
            'best_loss': self.best_loss,
            'total_steps': self.global_step,
            **training_summary  # Include coherence metrics from parent
        }

def main():
    parser = argparse.ArgumentParser(description='Train HGTD model with extensive logging')
    parser.add_argument('--config', default='cfg.yaml', help='Path to config file')
    parser.add_argument('--data_root', default=None, help='Override data root path')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--log_dir', default='logs', help='Directory for logs')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--experiment_name', default=None, help='Experiment name (default: timestamp)')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"hgtd_experiment_{timestamp}"
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    metrics_log_file = setup_logging(args.log_dir, args.experiment_name, log_level)
    tb_writer = setup_tensorboard(args.log_dir, args.experiment_name)
    
    # Log system information
    log_system_info()
    
    try:
        # Load configuration
        config_dict = load_config(args.config)
        
        # Override with command line arguments if provided
        if args.data_root:
            config_dict['data_root'] = args.data_root
            logging.info(f"Overriding data_root: {args.data_root}")
        if args.batch_size:
            config_dict['batch_size'] = args.batch_size
            logging.info(f"Overriding batch_size: {args.batch_size}")
        
        cfg = Config(config_dict)
        
        # Log configuration
        log_config_info(cfg, config_dict)
        
        # Initialize wandb with enhanced config
        wandb_config = config_dict.copy()
        wandb_config.update({
            'experiment_name': args.experiment_name,
            'log_dir': args.log_dir,
            'checkpoint_dir': args.checkpoint_dir,
            'resume': args.resume is not None
        })
        
        wandb.init(
            project="hgtd_tcga", 
            name=args.experiment_name,
            config=wandb_config,
            tags=['hgtd', 'multi-omics', 'diffusion']
        )
        
        # Create enhanced trainer
        trainer = EnhancedCurriculumTrainer(
            cfg, metrics_log_file, tb_writer, args.checkpoint_dir, args.experiment_name
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logging.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}, "
                        f"phase {checkpoint.get('phase', 'unknown')}")
        
        # Train the model
        final_metrics = trainer.train_all()
        
        # Log final summary
        logging.info("=== TRAINING SUMMARY ===")
        for key, value in final_metrics.items():
            logging.info(f"{key}: {value}")
        
        wandb.summary.update(final_metrics)
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if 'tb_writer' in locals():
            tb_writer.close()
        wandb.finish()
        logging.info("Training script completed.")

if __name__ == "__main__":
    main()