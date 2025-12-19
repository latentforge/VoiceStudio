import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, List
import wandb
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from models.tts_lora import TTSWithLoRA
from data_loading.dataset import create_dataloader
from utils.losses import CombinedLoss
from config import TrainingConfig


class TTSLoRATrainer:
    """
    Custom trainer for TTS with LoRA and Hypernetwork
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: TTSWithLoRA,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_function: Optional[CombinedLoss] = None
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Initialize loss function
        if loss_function is None:
            self.loss_function = CombinedLoss(sample_rate=config.sample_rate)
        else:
            self.loss_function = loss_function
        
        # Setup device
        self.device = torch.device(config.device if config.device != "auto" else 
                                  ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Move model to device
        self.model.to(self.device)
        self.loss_function.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            eta_min=1e-6
        )
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"tts_lora_{wandb.util.generate_id()}"
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Total training steps: {len(self.train_dataloader) * self.config.num_epochs}")
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_losses = self._train_epoch()
            
            # Validation phase
            if self.val_dataloader is not None:
                val_losses = self._validate_epoch()
                self._log_metrics(train_losses, val_losses, phase="validation")
                
                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self._save_checkpoint(is_best=True)
            else:
                self._log_metrics(train_losses, phase="training")
            
            # Save regular checkpoint
            if (epoch + 1) % (self.config.save_steps // len(self.train_dataloader)) == 0:
                self._save_checkpoint()
        
        print("Training completed!")
        if self.config.use_wandb:
            wandb.finish()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            losses = self._training_step(batch)
            
            # Backward pass
            self.scaler.scale(losses['total']).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Scheduler step
            self.scheduler.step()
            
            # Update global step
            self.global_step += 1
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
            
            # Log step metrics
            if self.global_step % self.config.logging_steps == 0:
                step_metrics = {f"train/{k}": v.item() for k, v in losses.items()}
                step_metrics['train/learning_rate'] = current_lr
                step_metrics['train/global_step'] = self.global_step
                
                if self.config.use_wandb:
                    wandb.log(step_metrics, step=self.global_step)
        
        # Average losses over epoch
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        return avg_losses
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {}
        
        progress_bar = tqdm(self.val_dataloader, desc=f"Validation Epoch {self.epoch + 1}")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                losses = self._validation_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{losses['total'].item():.4f}"
                })
        
        # Average losses over epoch
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        return avg_losses
    
    def _training_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single training step"""
        # Extract data from batch
        audio = batch['audio']  # [batch_size, audio_length]
        texts = batch['text']  # List of strings
        speaker_descriptions = batch['speaker_descriptions']  # List of strings
        
        # Use reference audio as speaker characteristics
        speaker_audio = audio  # Same audio as reference
        
        # Forward pass through model
        with torch.cuda.amp.autocast():
            generated_audio = self.model(
                content_text=texts,
                speaker_audio=speaker_audio,
                speaker_text=speaker_descriptions,
                sample_rate=self.config.sample_rate
            )
            
            # Compute losses
            losses = self.loss_function(
                generated_audio=generated_audio,
                target_audio=audio,
                reference_audio=speaker_audio
            )
        
        return losses
    
    def _validation_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single validation step"""
        # Same as training step but without gradient computation
        return self._training_step(batch)
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def _log_metrics(
        self, 
        train_losses: Dict[str, float], 
        val_losses: Optional[Dict[str, float]] = None,
        phase: str = "training"
    ):
        """Log metrics to console and wandb"""
        # Log to console
        print(f"\n{phase.capitalize()} Metrics:")
        for key, value in train_losses.items():
            print(f"  Train {key}: {value:.4f}")
        
        if val_losses is not None:
            for key, value in val_losses.items():
                print(f"  Val {key}: {value:.4f}")
        
        # Log to wandb
        if self.config.use_wandb:
            log_dict = {}
            
            # Training metrics
            for key, value in train_losses.items():
                log_dict[f"epoch/train_{key}"] = value
            
            # Validation metrics
            if val_losses is not None:
                for key, value in val_losses.items():
                    log_dict[f"epoch/val_{key}"] = value
            
            log_dict['epoch'] = self.epoch
            wandb.log(log_dict, step=self.global_step)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")


def create_trainer(
    config: TrainingConfig,
    model: Optional[TTSWithLoRA] = None,
    train_dataloader: Optional[DataLoader] = None,
    val_dataloader: Optional[DataLoader] = None
) -> TTSLoRATrainer:
    """
    Create trainer with default configurations
    """
    # Create model if not provided
    if model is None:
        from peft import LoraConfig, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules
        )
        
        hypernetwork_config = {
            'hidden_dim': config.hypernetwork_hidden_dim,
            'num_layers': config.hypernetwork_num_layers
        }
        
        model = TTSWithLoRA(
            tts_model_name=config.tts_model_name,
            clap_model_name=config.clap_model_name,
            qwen_model_name=config.qwen_model_name,
            lora_config=lora_config,
            hypernetwork_config=hypernetwork_config,
            use_qwen=False  # Can be made configurable
        )
    
    # Create dataloaders if not provided
    if train_dataloader is None:
        train_dataloader = create_dataloader(
            dataset_name=config.dataset_name,
            split="train",
            batch_size=config.batch_size,
            sample_rate=config.sample_rate,
            cache_dir=config.cache_dir
        )
    
    if val_dataloader is None:
        try:
            val_dataloader = create_dataloader(
                dataset_name=config.dataset_name,
                split="validation",
                batch_size=config.batch_size,
                sample_rate=config.sample_rate,
                cache_dir=config.cache_dir,
                shuffle=False
            )
        except:
            # If validation split doesn't exist, use a subset of training data
            print("Validation split not found, using subset of training data")
            val_dataloader = None
    
    # Create loss function
    loss_function = CombinedLoss(sample_rate=config.sample_rate)
    
    # Create trainer
    trainer = TTSLoRATrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function
    )
    
    return trainer
