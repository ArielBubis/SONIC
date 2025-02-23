import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 
class ShallowEmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, precomputed_item_embeddings=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Just the embeddings - no additional layers
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Initialize with precomputed embeddings if provided
        if precomputed_item_embeddings is not None:
            with torch.no_grad():
                self.item_embeddings.weight.copy_(torch.tensor(precomputed_item_embeddings))
        
        # Cosine similarity for scoring
        self.similarity = nn.CosineSimilarity(dim=1)
        
        self.to(self.device)
        
    def forward(self, user_indices, item_indices):
        """Forward pass - just embeddings and similarity"""
        # Get embeddings
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)
        
        # Compute similarity directly
        return self.similarity(user_embeds, item_embeds)
    
    def extract_embeddings(self, normalize=True):
        """Extract embeddings for inference"""
        self.eval()
        with torch.no_grad():
            user_embeddings = self.user_embeddings.weight.cpu().numpy()
            item_embeddings = self.item_embeddings.weight.cpu().numpy()
            
            if normalize:
                user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
                item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
            
            return user_embeddings, item_embeddings