import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm

class ShallowEmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, 
                 precomputed_user_embeddings=None, emb_dim_out=300):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_dim_in = emb_dim_in

        # Use half precision for embeddings
        dtype = torch.float32 if self.device.type == 'cuda' else torch.float32

        # Initialize embeddings with optimized settings
        if precomputed_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in, dtype=dtype)
        else:
            precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings).to(dtype)
            self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings, freeze=True)

        if precomputed_item_embeddings is None:
            self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in, dtype=dtype)
        else:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings).to(dtype)
            self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings, freeze=True)

        # Optimize model architecture
        self.model = nn.Sequential(
            nn.Linear(self.emb_dim_in, emb_dim_out, dtype=dtype),
            nn.ReLU(inplace=True)  # Use inplace operations
        ).to(self.device)

        self.cossim = nn.CosineSimilarity(dim=1)
        
    def forward(self, user_indices, item_indices):
        # Using torch.no_grad for inference speedup when possible
        with torch.set_grad_enabled(self.training):
            user_embeds = self.user_embeddings(user_indices)
            item_embeds = self.item_embeddings(item_indices)
            
            user_embeds = self.model(user_embeds)
            item_embeds = self.model(item_embeds)
            
            return self.cossim(user_embeds, item_embeds)

    def train_model(self, train_loader, val_loader, num_epochs=100, neg_samples=20,
                   patience_threshold=16, l2=0, use_confidence=False):
        """Optimized training loop"""
        print(f"Training on {self.device}")
        
        # Initialize mixed precision training
        scaler = torch.amp.GradScaler()
        
        # Optimize memory allocation
        torch.backends.cudnn.benchmark = True
        
        # Initialize optimizer with larger learning rate and momentum
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=l2,
            betas=(0.9, 0.999)
        )
        
        # Use cyclic learning rate for faster convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader)
        )

        patience_counter = 0
        best_val_loss = float('inf')
        
        # Pre-allocate tensors for negative sampling
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_user, batch_pos_item, batch_confidence in progress_bar:
                batch_size = len(batch_user)
                
                # Move data to device and generate negative samples efficiently
                batch_user = batch_user.to(self.device, non_blocking=True)
                batch_pos_item = batch_pos_item.to(self.device, non_blocking=True)
                batch_neg_items = torch.randint(
                    0, self.item_embeddings.weight.size(0),
                    (batch_size, neg_samples),
                    device=self.device
                )
                
                # Optimize confidence score calculation
                if use_confidence:
                    batch_confidence = batch_confidence.to(self.device, non_blocking=True)
                    batch_confidence = (1 + 2 * torch.log1p(batch_confidence))
                else:
                    batch_confidence = torch.ones(batch_size, device=self.device)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Use automatic mixed precision
                with autocast():
                    # Compute all scores in a single forward pass
                    pos_score = self(batch_user, batch_pos_item).unsqueeze(1)
                    neg_scores = torch.stack([
                        self(batch_user, batch_neg_items[:, i])
                        for i in range(neg_samples)
                    ], dim=1)
                    
                    # Vectorized loss computation
                    loss = torch.clamp(0.2 - pos_score + neg_scores, min=0)
                    loss = (loss * batch_confidence.unsqueeze(1)).mean()
                    
                    if l2 > 0:
                        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
                        loss = loss + l2 * l2_loss

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                current_loss = loss.item()
                total_train_loss += current_loss
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

            # Quick validation phase
            val_loss = self._validate(val_loader, neg_samples, use_confidence)
            
            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {total_train_loss/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_threshold:
                    print("Early stopping triggered")
                    break
        
        return best_val_loss

    @torch.no_grad()
    def _validate(self, val_loader, neg_samples, use_confidence):
        """Optimized validation"""
        self.eval()
        total_val_loss = 0
        
        for batch_user, batch_pos_item, batch_confidence in val_loader:
            batch_size = len(batch_user)
            
            batch_user = batch_user.to(self.device, non_blocking=True)
            batch_pos_item = batch_pos_item.to(self.device, non_blocking=True)
            batch_neg_items = torch.randint(
                0, self.item_embeddings.weight.size(0),
                (batch_size, neg_samples),
                device=self.device
            )
            
            if use_confidence:
                batch_confidence = batch_confidence.to(self.device, non_blocking=True)
                batch_confidence = (1 + 2 * torch.log1p(batch_confidence))
            else:
                batch_confidence = torch.ones(batch_size, device=self.device)
            
            with autocast():
                pos_score = self(batch_user, batch_pos_item).unsqueeze(1)
                neg_scores = torch.stack([
                    self(batch_user, batch_neg_items[:, i])
                    for i in range(neg_samples)
                ], dim=1)
                
                loss = torch.clamp(0.2 - pos_score + neg_scores, min=0)
                loss = (loss * batch_confidence.unsqueeze(1)).mean()
            
            total_val_loss += loss.item()
        
        return total_val_loss / len(val_loader)

    @torch.no_grad()
    def extract_embeddings(self, normalize=True, batch_size=2048):
        """Optimized embedding extraction"""
        self.eval()
        
        def process_embeddings(embeddings, desc):
            num_embeddings = len(embeddings)
            processed = []
            
            for i in range(0, num_embeddings, batch_size):
                batch = embeddings[i:i + batch_size].to(self.device)
                with torch.amp.autocast():
                    processed.append(self.model(batch).cpu().numpy())
            
            return np.concatenate(processed)

        user_embeddings = process_embeddings(
            self.user_embeddings.weight.data,
            "Processing user embeddings"
        )
        
        item_embeddings = process_embeddings(
            self.item_embeddings.weight.data,
            "Processing item embeddings"
        )

        if normalize:
            user_embeddings /= np.linalg.norm(user_embeddings, axis=1, keepdims=True)
            item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        
        return user_embeddings, item_embeddings