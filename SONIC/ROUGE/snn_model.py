# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np 
# class ShallowEmbeddingModel(nn.Module):
#     def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, emb_dim_out=300):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Initialize embeddings
#         self.user_embeddings = nn.Embedding(num_users, emb_dim_in)
#         self.item_embeddings = nn.Embedding(num_items, emb_dim_in)
#         if precomputed_item_embeddings is not None:
#             with torch.no_grad():
#                 self.item_embeddings.weight.copy_(torch.tensor(precomputed_item_embeddings))
#         # Simple model architecture
#         self.model = nn.Sequential(
#             nn.Linear(emb_dim_in, emb_dim_out),
#             nn.ReLU()
#         )
        
#         # Cosine similarity for scoring
#         self.cossim = nn.CosineSimilarity(dim=1)
        
#         # Move model to device
#         self.to(self.device)
        
#     def forward(self, user_indices, item_indices):
#         """Forward pass"""
#         # Get embeddings
#         user_embeds = self.user_embeddings(user_indices)
#         item_embeds = self.item_embeddings(item_indices)
        
#         # Pass through model
#         user_embeds = self.model(user_embeds)
#         item_embeds = self.model(item_embeds)
        
#         # Calculate similarity
#         return self.cossim(user_embeds, item_embeds)
    
#     def train_model(self, train_loader, val_loader, run_name, num_epochs=100, neg_samples=20,
#                 patience_threshold=16, l2=0, use_confidence=False,):
#         """Enhanced training loop with early stopping and L2 regularization"""
#         optimizer = torch.optim.Adam(self.parameters())
#         best_val_loss = float('inf')
#         patience_counter = 0
        
#         for epoch in range(num_epochs):
#             # Training phase
#             self.train()
#             total_train_loss = 0
            
#             for batch_user, batch_pos_item, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
#                 batch_size = len(batch_user)
                
#                 batch_user = batch_user.to(self.device)
#                 batch_pos_item = batch_pos_item.to(self.device)
                
#                 batch_neg_items = torch.randint(
#                     0, self.item_embeddings.weight.size(0),
#                     (batch_size, neg_samples),
#                     device=self.device
#                 )
                
#                 optimizer.zero_grad()
                
#                 pos_score = self(batch_user, batch_pos_item).unsqueeze(1)
                
#                 neg_scores = []
#                 for i in range(neg_samples):
#                     neg_score = self(batch_user, batch_neg_items[:, i])
#                     neg_scores.append(neg_score.unsqueeze(1))
#                 neg_scores = torch.cat(neg_scores, dim=1)
                
#                 # Base loss
#                 loss = torch.clamp(0.2 - pos_score + neg_scores, min=0).mean()
                
#                 # Add L2 regularization if specified
#                 if l2 > 0:
#                     l2_loss = 0
#                     for param in self.parameters():
#                         l2_loss += torch.norm(param)
#                     loss += l2 * l2_loss
                
#                 loss.backward()
#                 optimizer.step()
                
#                 total_train_loss += loss.item()
            
#             avg_train_loss = total_train_loss / len(train_loader)
            
#             # Validation phase
#             avg_val_loss = self._validate(val_loader, neg_samples, l2)
            
#             print(f"\nEpoch {epoch+1}/{num_epochs}")
#             print(f"Train Loss: {avg_train_loss:.4f}")
#             print(f"Val Loss: {avg_val_loss:.4f}")
            
#             # Save best model and check early stopping
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': self.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': best_val_loss,
#                 }, f'checkpoints/{run_name}_best.pt')
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience_threshold:
#                     print(f"Early stopping triggered after {epoch + 1} epochs")
#                     break
        
#         return best_val_loss
#     def _validate(self, val_loader, neg_samples, l2):
#         """Separate validation method for cleaner code"""
#         self.eval()
#         total_val_loss = 0
        
#         with torch.no_grad():
#             for batch_user, batch_pos_item, _ in val_loader:
#                 batch_size = len(batch_user)
                
#                 batch_user = batch_user.to(self.device)
#                 batch_pos_item = batch_pos_item.to(self.device)
#                 batch_neg_items = torch.randint(
#                     0, self.item_embeddings.weight.size(0),
#                     (batch_size, neg_samples),
#                     device=self.device
#                 )
                
#                 pos_score = self(batch_user, batch_pos_item).unsqueeze(1)
                
#                 neg_scores = []
#                 for i in range(neg_samples):
#                     neg_score = self(batch_user, batch_neg_items[:, i])
#                     neg_scores.append(neg_score.unsqueeze(1))
#                 neg_scores = torch.cat(neg_scores, dim=1)
                
#                 loss = torch.clamp(0.2 - pos_score + neg_scores, min=0).mean()
                
#                 if l2 > 0:
#                     l2_loss = 0
#                     for param in self.parameters():
#                         l2_loss += torch.norm(param)
#                     loss += l2 * l2_loss
                
#                 total_val_loss += loss.item()
        
#         return total_val_loss / len(val_loader)

#     def extract_embeddings(self, normalize=True):
#         """Extract embeddings for inference"""
#         self.eval()
#         with torch.no_grad():
#             # Get embeddings
#             user_embeddings = self.user_embeddings.weight.data
#             item_embeddings = self.item_embeddings.weight.data
            
#             # Pass through model
#             user_embeddings = self.model(user_embeddings).cpu().numpy()
#             item_embeddings = self.model(item_embeddings).cpu().numpy()
            
#             # Normalize if requested
#             if normalize:
#                 user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
#                 item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
            
#             return user_embeddings, item_embeddings