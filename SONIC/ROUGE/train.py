# from dataclasses import dataclass
# import torch
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import os
# from tqdm.auto import tqdm
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from typing import Dict, Any

# @dataclass
# class TrainingConfig:
#     """Configuration for model training."""
#     model_name: str = "BERT4Rec"
#     max_seq_len: int = 128
#     batch_size: int = 128
#     num_epochs: int = 200
#     lr: float = 0.001
#     weight_decay: float = 0.01
#     patience_threshold: int = 16
#     grad_clip_norm: float = 1.0
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# class ModelTrainer:
#     """Generic trainer class for recommendation models."""
#     def __init__(
#         self,
#         model: nn.Module,
#         config: TrainingConfig,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         criterion: nn.Module = None
#     ):
#         self.model = model.to(config.device)
#         self.config = config
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=-100).to(config.device)
        
#         self.optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=config.lr,
#             weight_decay=config.weight_decay
#         )
#         self.scheduler = CosineAnnealingLR(
#             self.optimizer,
#             T_max=config.num_epochs,
#             eta_min=1e-6
#         )
#         self.scaler = torch.cuda.amp.GradScaler()
        
#         self.best_val_loss = float('inf')
#         self.patience_counter = 0

#     def train_epoch(self):
#         """Train for one epoch."""
#         self.model.train()
#         total_loss = 0
        
#         for batch in self.train_loader:
#             loss = self._train_step(batch)
#             total_loss += loss
            
#         return total_loss / len(self.train_loader)

#     def _train_step(self, batch):
#         """Single training step."""
#         input_ids = batch['input_ids'].to(self.config.device)
#         labels = batch['labels'].to(self.config.device)
#         attention_mask = batch['attention_mask'].to(self.config.device)
        
#         self.optimizer.zero_grad()
        
#         with torch.cuda.amp.autocast():
#             outputs = self.model(input_ids, attention_mask)
#             loss = self.criterion(
#                 outputs.view(-1, outputs.size(-1)),
#                 labels.view(-1)
#             )
            
#         self.scaler.scale(loss).backward()
#         torch.nn.utils.clip_grad_norm_(
#             self.model.parameters(),
#             max_norm=self.config.grad_clip_norm
#         )
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
        
#         return loss.detach().cpu().item()

#     def evaluate(self):
#         """Evaluate the model."""
#         self.model.eval()
#         total_loss = 0
        
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 loss = self._eval_step(batch)
#                 total_loss += loss
                
#         return total_loss / len(self.val_loader)

#     def _eval_step(self, batch):
#         """Single evaluation step."""
#         input_ids = batch['input_ids'].to(self.config.device)
#         labels = batch['labels'].to(self.config.device)
#         attention_mask = batch['attention_mask'].to(self.config.device)
        
#         outputs = self.model(input_ids, attention_mask)
#         loss = self.criterion(
#             outputs.view(-1, outputs.size(-1)),
#             labels.view(-1)
#         )
        
#         return loss.item()

#     def train(self):
#         """Full training loop."""
#         os.makedirs(f'checkpoints/{self.config.model_name}', exist_ok=True)
        
#         for epoch in tqdm(range(self.config.num_epochs)):
#             train_loss = self.train_epoch()
#             val_loss = self.evaluate()
            
#             self.scheduler.step()
            
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 self._save_checkpoint(epoch, val_loss)
#                 self.patience_counter = 0
#             else:
#                 self.patience_counter += 1
                
#             if self.patience_counter >= self.config.patience_threshold:
#                 print("Early stopping triggered.")
#                 break
        
#         return self.best_val_loss

#     def _save_checkpoint(self, epoch, val_loss):
#         """Save model checkpoint."""
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': val_loss,
#         }
#         torch.save(
#             checkpoint,
#             f'checkpoints/{self.config.model_name}/checkpoint_{epoch}.pt'
#         )