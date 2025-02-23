import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ShallowEmbeddingModel(nn.Module):
    """
    A shallow embedding model for user-item interactions using cosine similarity.
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        emb_dim_in (int): Input embedding dimension
        precomputed_item_embeddings (np.ndarray, optional): Precomputed item embeddings
        precomputed_user_embeddings (np.ndarray, optional): Precomputed user embeddings
        emb_dim_out (int, optional): Output embedding dimension. Defaults to 300
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
        """
        Forward pass of the model.
        
        Args:
            user_indices (torch.Tensor): User indices
            item_indices (torch.Tensor): Item indices
            
        Returns:
            torch.Tensor: Cosine similarity scores
        """
        # Get embeddings
        user_embeds = self.user_embeddings(user_indices).to(self.device)
        item_embeds = self.item_embeddings(item_indices).to(self.device)

        # Transform embeddings
        user_embeds = self.model(user_embeds)
        item_embeds = self.model(item_embeds)

        # Calculate cosine similarity
        scores = self.cossim(user_embeds, item_embeds)
        return scores

    def extract_embeddings(self, normalize=True):
        """
        Extract user and item embeddings.
        
        Args:
            normalize (bool): Whether to normalize embeddings. Defaults to True.
            
        Returns:
            tuple: (user_embeddings, item_embeddings) as numpy arrays
        """
        user_embeddings = self.user_embeddings.weight.data
        item_embeddings = self.item_embeddings.weight.data

        with torch.no_grad():
            user_embeddings = self.model(user_embeddings.to(self.device)).cpu().numpy()
            item_embeddings = self.model(item_embeddings.to(self.device)).cpu().numpy()

        if normalize:
            user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
            item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        return user_embeddings, item_embeddings

    def train_model(self, train_loader, val_loader, num_epochs=100, neg_samples=20, 
                   patience_threshold=16, l2=0, use_confidence=False, device=None,
                   writer=None, checkpoint_dir=None, run_name=None):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            neg_samples (int): Number of negative samples per positive
            patience_threshold (int): Early stopping patience
            l2 (float): L2 regularization weight
            use_confidence (bool): Whether to use confidence scores
            device (torch.device): Device to train on
            writer (SummaryWriter): TensorBoard writer
            checkpoint_dir (str): Directory to save checkpoints
            run_name (str): Name of the current run
            
        Returns:
            float: Best validation loss achieved
        """
        if device is None:
            device = self.device

        def hinge_loss(y_pos, y_neg, confidence, dlt=0.2):
            """
            Calculate hinge loss with proper broadcasting
            
            Args:
                y_pos: Tensor of shape (batch_size, neg_samples)
                y_neg: Tensor of shape (batch_size, neg_samples)
                confidence: Tensor of shape (batch_size, neg_samples)
            """
            # Ensure all inputs have the same shape
            assert y_pos.shape == y_neg.shape == confidence.shape
            
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
                print(f"Checkpoint saved to {path}")

        patience_counter = 0
        best_val_loss = float('inf')
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            total_train_loss = 0

            for batch_user, batch_pos_item, batch_confidence in train_loader:
                batch_size = len(batch_user)
                
                # Prepare batch data for negative sampling
                batch_user_expanded = batch_user.unsqueeze(1).repeat(1, neg_samples).reshape(-1).to(device)
                batch_pos_item = batch_pos_item.to(device)
                
                # Generate negative samples
                batch_neg_items = torch.randint(0, self.item_embeddings.weight.size(0), 
                                              (batch_size, neg_samples)).to(device)
                
                # Handle confidence scores
                if not use_confidence:
                    batch_confidence = torch.ones_like(batch_confidence)
                batch_confidence = batch_confidence.to(device)
                
                if use_confidence:
                    batch_confidence = (1 + 2 * torch.log(1 + batch_confidence))
                
                # Expand confidence scores for negative samples
                batch_confidence = batch_confidence.unsqueeze(1).repeat(1, neg_samples)
                
                optimizer.zero_grad()
                
                # Forward pass for positive samples
                pos_score = self(batch_user, batch_pos_item)
                pos_score = pos_score.unsqueeze(1).repeat(1, neg_samples)
                
                # Forward pass for negative samples
                neg_scores = []
                for i in range(neg_samples):
                    neg_score = self(batch_user, batch_neg_items[:, i])
                    neg_scores.append(neg_score.unsqueeze(1))
                neg_scores = torch.cat(neg_scores, dim=1)
                
                # Calculate loss ensuring all tensors have the same shape
                loss = hinge_loss(pos_score, neg_scores, batch_confidence)
                if l2 > 0:
                    l2_loss = sum(torch.sum(param ** 2) for param in self.parameters())
                    loss = loss + l2 * l2_loss

                # Backward pass
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            if writer:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)

            # Validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_user, batch_pos_item, batch_confidence in val_loader:
                    batch_user = batch_user.repeat_interleave(neg_samples).to(device)
                    batch_pos_item = batch_pos_item.repeat_interleave(neg_samples).to(device)
                    batch_neg_items = torch.randint(0, self.item_embeddings.weight.size(0), 
                                                 (len(batch_user),)).to(device)

                    if not use_confidence:
                        batch_confidence = torch.ones_like(batch_confidence)
                    batch_confidence = batch_confidence.repeat_interleave(neg_samples).to(device)
                    if use_confidence:
                        batch_confidence = (1 + 2 * torch.log(1 + batch_confidence))

                    pos_score = self(batch_user, batch_pos_item)
                    neg_scores = self(batch_user, batch_neg_items)

                    loss = hinge_loss(pos_score, neg_scores, batch_confidence)
                    if l2 > 0:
                        l2_loss = sum(torch.sum(param ** 2) for param in self.parameters())
                        loss = loss + l2 * l2_loss

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            if writer:
                writer.add_scalar('Loss/val', avg_val_loss, epoch)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                print(f'New best at epoch {epoch}')
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

            if patience_counter >= patience_threshold:
                print('Applying early stop')
                break

        return best_val_loss