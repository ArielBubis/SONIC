import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Any, Optional
import numpy as np

class BERT4Rec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        bert_config: Dict[str, Any],
        precomputed_item_embeddings: Optional[np.ndarray] = None,
        add_head: bool = True,
        tie_weights: bool = True,
        padding_idx: int = -1,
        init_std: float = 0.02
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        # Initialize BERT model first
        bert_config['vocab_size'] = vocab_size  # Ensure BERT config matches vocab size
        self.transformer_model = BertModel(BertConfig(**bert_config))

        if precomputed_item_embeddings is not None:
            input_dim = precomputed_item_embeddings.shape[1]
            hidden_size = bert_config['hidden_size']
            
            # Create projection layer if dimensions don't match
            if input_dim != hidden_size:
                self.embedding_projection = nn.Linear(input_dim, hidden_size)
                precomputed_embeddings = torch.from_numpy(
                    precomputed_item_embeddings.astype(np.float32)
                )
                with torch.no_grad():
                    precomputed_embeddings = self.embedding_projection(precomputed_embeddings)
            else:
                precomputed_embeddings = torch.from_numpy(
                    precomputed_item_embeddings.astype(np.float32)
                )
            
            # Pad embeddings if needed
            if precomputed_embeddings.shape[0] < vocab_size:
                padding = torch.zeros(
                    vocab_size - precomputed_embeddings.shape[0], 
                    hidden_size,
                    dtype=precomputed_embeddings.dtype
                )
                precomputed_embeddings = torch.cat([precomputed_embeddings, padding], dim=0)
            
            # Initialize embeddings with precomputed values
            self.item_embeddings = nn.Embedding.from_pretrained(
                precomputed_embeddings,
                padding_idx=padding_idx,
                freeze=False
            )
            
            # Copy item embeddings to BERT embeddings
            with torch.no_grad():
                self.transformer_model.embeddings.word_embeddings.weight.copy_(
                    self.item_embeddings.weight
                )
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=bert_config['hidden_size'],
                padding_idx=padding_idx
            )
            # Copy embeddings to BERT
            with torch.no_grad():
                self.transformer_model.embeddings.word_embeddings.weight.copy_(
                    self.item_embeddings.weight
                )

        if self.add_head:
            self.head = nn.Linear(
                bert_config['hidden_size'],
                vocab_size,
                bias=True  # Changed to match BERT's output layer
            )
            if self.tie_weights:
                self.head.weight = self.item_embeddings.weight

        # Initialize or zero padding token
        if precomputed_item_embeddings is None:
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.item_embeddings(input_ids)
        
        # Apply projection if needed
        if hasattr(self, 'embedding_projection'):
            embeds = self.embedding_projection(embeds)
            
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs