import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ShallowEmbeddingModel(nn.Module):
    """
    A shallow embedding model for user-item interactions using cosine similarity.
    """
    def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, 
                 precomputed_user_embeddings=None, emb_dim_out=300):
        super(ShallowEmbeddingModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_dim_in = emb_dim_in

        # Initialize user embeddings
        if precomputed_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in)
        else:
            precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings)
            assert precomputed_user_embeddings.size(1) == emb_dim_in
            self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings)

        # Initialize item embeddings
        if precomputed_item_embeddings is None:
            self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in)
        else:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings)
            assert precomputed_item_embeddings.size(1) == emb_dim_in
            self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings)

        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(self.emb_dim_in, emb_dim_out),
            nn.ReLU()
        )

        self.model.to(self.device)
        self.cossim = torch.nn.CosineSimilarity()

    def freeze_item_embs(self, flag):
        """Freeze or unfreeze item embeddings"""
        self.item_embeddings.weight.requires_grad = not flag

    def freeze_user_embs(self, flag):
        """Freeze or unfreeze user embeddings"""
        self.user_embeddings.weight.requires_grad = not flag

    def forward(self, user_indices, item_indices):
        """Forward pass of the model"""
        user_embeds = self.user_embeddings(user_indices).to(self.device)
        item_embeds = self.item_embeddings(item_indices).to(self.device)

        user_embeds = self.model(user_embeds)
        item_embeds = self.model(item_embeds)

        scores = self.cossim(user_embeds, item_embeds)
        return scores

    def train_model(self, train_loader, val_loader, num_epochs=100, neg_samples=20, 
                   patience_threshold=16, l2=0, use_confidence=False, device=None,
                   writer=None, checkpoint_dir=None, run_name=None):
        """Train the model"""
        if device is None:
            device = self.device
            
        print(f"Training on device: {device}")
        print(f"Training parameters:")
        print(f"- Number of epochs: {num_epochs}")
        print(f"- Negative samples: {neg_samples}")
        print(f"- Patience threshold: {patience_threshold}")
        print(f"- L2 regularization: {l2}")
        print(f"- Using confidence: {use_confidence}")

        def hinge_loss(y_pos, y_neg, confidence, dlt=0.2):
            """Calculate hinge loss with proper broadcasting"""
            assert y_pos.shape == y_neg.shape == confidence.shape, \
                f"Shape mismatch: pos={y_pos.shape}, neg={y_neg.shape}, conf={confidence.shape}"
            loss = dlt - y_pos + y_neg
            loss = torch.clamp(loss, min=0) * confidence
            return torch.mean(loss)

        def save_checkpoint(epoch, loss):
            if checkpoint_dir and run_name:
                path = f'{checkpoint_dir}/{run_name}_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, path)
                print(f"📝 Checkpoint saved to {path}")

        patience_counter = 0
        best_val_loss = float('inf')
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        print("\nStarting training...")
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            batch_count = 0
            
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            progress_bar = tqdm(train_loader, desc='Training')
            
            for batch_user, batch_pos_item, batch_confidence in progress_bar:
                batch_size = len(batch_user)
                batch_count += 1
                
                # Move data to device
                batch_user = batch_user.to(device)
                batch_pos_item = batch_pos_item.to(device)
                
                # Generate negative samples
                batch_neg_items = torch.randint(0, self.item_embeddings.weight.size(0), 
                                              (batch_size, neg_samples)).to(device)
                
                # Handle confidence scores
                if not use_confidence:
                    batch_confidence = torch.ones(batch_size, device=device)
                else:
                    batch_confidence = batch_confidence.to(device)
                    batch_confidence = (1 + 2 * torch.log(1 + batch_confidence))
                
                # Reshape confidence scores to match dimensions
                batch_confidence = batch_confidence.view(-1, 1).expand(-1, neg_samples)
                
                optimizer.zero_grad()
                
                # Forward pass for positive samples
                pos_score = self(batch_user, batch_pos_item)
                pos_score = pos_score.view(-1, 1).expand(-1, neg_samples)
                
                # Forward pass for negative samples
                neg_scores = []
                for i in range(neg_samples):
                    neg_score = self(batch_user, batch_neg_items[:, i])
                    neg_scores.append(neg_score.view(-1, 1))
                neg_scores = torch.cat(neg_scores, dim=1)  # Shape: [batch_size, neg_samples]
                
                # Calculate loss
                try:
                    loss = hinge_loss(pos_score, neg_scores, batch_confidence)
                    if l2 > 0:
                        l2_loss = sum(torch.sum(param ** 2) for param in self.parameters())
                        loss = loss + l2 * l2_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    current_loss = loss.item()
                    
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'avg_loss': f'{total_train_loss/batch_count:.4f}'
                    })
                    
                except RuntimeError as e:
                    print(f"\n❌ Error in batch {batch_count}:")
                    print(e)
                    raise e

            avg_train_loss = total_train_loss / len(train_loader)
            if writer:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
            print(f"\n🏋️ Training Loss: {avg_train_loss:.4f}")

            # Validation phase
            self.eval()
            total_val_loss = 0
            
            print("\nStarting validation...")
            progress_bar = tqdm(val_loader, desc='Validation')
            
            with torch.no_grad():
                for batch_user, batch_pos_item, batch_confidence in progress_bar:
                    batch_size = len(batch_user)
                    
                    # Move data to device
                    batch_user = batch_user.to(device)
                    batch_pos_item = batch_pos_item.to(device)
                    
                    # Generate negative samples
                    batch_neg_items = torch.randint(0, self.item_embeddings.weight.size(0), 
                                                  (batch_size, neg_samples)).to(device)
                    
                    # Handle confidence scores
                    if not use_confidence:
                        batch_confidence = torch.ones(batch_size, device=device)
                    else:
                        batch_confidence = batch_confidence.to(device)
                        batch_confidence = (1 + 2 * torch.log(1 + batch_confidence))
                    
                    # Reshape confidence scores
                    batch_confidence = batch_confidence.view(-1, 1).expand(-1, neg_samples)
                    
                    # Forward pass for positive samples
                    pos_score = self(batch_user, batch_pos_item)
                    pos_score = pos_score.view(-1, 1).expand(-1, neg_samples)
                    
                    # Forward pass for negative samples
                    neg_scores = []
                    for i in range(neg_samples):
                        neg_score = self(batch_user, batch_neg_items[:, i])
                        neg_scores.append(neg_score.view(-1, 1))
                    neg_scores = torch.cat(neg_scores, dim=1)
                    
                    loss = hinge_loss(pos_score, neg_scores, batch_confidence)
                    if l2 > 0:
                        l2_loss = sum(torch.sum(param ** 2) for param in self.parameters())
                        loss = loss + l2 * l2_loss
                        
                    total_val_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_val_loss = total_val_loss / len(val_loader)
            if writer:
                writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f"\n📊 Validation Loss: {avg_val_loss:.4f}")

            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"📉 Learning rate adjusted: {old_lr:.6f} -> {new_lr:.6f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                print(f"🌟 New best validation loss: {avg_val_loss:.4f} (previous: {best_val_loss:.4f})")
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_checkpoint(epoch, best_val_loss)
                if checkpoint_dir and run_name:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, f'{checkpoint_dir}/{run_name}_best.pt')
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs")

            if patience_counter >= patience_threshold:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break

        print("\n✅ Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        return best_val_loss

    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load model checkpoint"""
        print(f"\n📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"✅ Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss

    def extract_embeddings(self, normalize=True, batch_size=1024):
        """Extract user and item embeddings"""
        print("\n🔄 Extracting embeddings...")
        self.eval()
        
        def process_embeddings_in_batches(embeddings, desc):
            num_embeddings = len(embeddings)
            processed_embeddings = []
            
            for i in tqdm(range(0, num_embeddings, batch_size), desc=desc):
                batch = embeddings[i:i + batch_size].to(self.device)
                with torch.no_grad():
                    processed = self.model(batch).cpu().numpy()
                processed_embeddings.append(processed)
            
            return np.concatenate(processed_embeddings, axis=0)
        
        # Process user embeddings
        user_embeddings = process_embeddings_in_batches(
            self.user_embeddings.weight.data,
            "Processing user embeddings"
        )
        
        # Process item embeddings
        item_embeddings = process_embeddings_in_batches(
            self.item_embeddings.weight.data,
            "Processing item embeddings"
        )
        
        if normalize:
            print("Normalizing embeddings...")
            user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
            item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        
        print(f"✅ Extracted embeddings:")
        print(f"- User embeddings shape: {user_embeddings.shape}")
        print(f"- Item embeddings shape: {item_embeddings.shape}")
        
        return user_embeddings, item_embeddings