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
        self.transformer_model = BertModel(BertConfig(**bert_config))
        print(f"Initialized BERT model with config: {bert_config}")

        if precomputed_item_embeddings is not None:
            input_dim = precomputed_item_embeddings.shape[1]
            hidden_size = bert_config['hidden_size']
            print(f"Precomputed embeddings shape: {precomputed_item_embeddings.shape}")

            # Create projection layer if dimensions don't match
            if input_dim != hidden_size:
                self.embedding_projection = nn.Linear(input_dim, hidden_size)
                precomputed_item_embeddings = torch.from_numpy(
                    precomputed_item_embeddings.astype(np.float32)
                )
                with torch.no_grad():
                    precomputed_item_embeddings = self.embedding_projection(precomputed_item_embeddings)
                print(f"Projected embeddings shape: {precomputed_item_embeddings.shape}")
            else:
                precomputed_item_embeddings = torch.from_numpy(
                    precomputed_item_embeddings.astype(np.float32)
                )

            # Pad embeddings if needed
            if precomputed_item_embeddings.shape[0] < vocab_size:
                padding = torch.zeros(
                    vocab_size - precomputed_item_embeddings.shape[0], 
                    hidden_size,
                    dtype=precomputed_item_embeddings.dtype
                )
                precomputed_item_embeddings = torch.cat([precomputed_item_embeddings, padding], dim=0)
                print(f"Padded embeddings shape: {precomputed_item_embeddings.shape}")

            # Initialize embeddings with precomputed values
            self.item_embeddings = nn.Embedding.from_pretrained(
                precomputed_item_embeddings,
                padding_idx=padding_idx,
                freeze=False
            )
            print(f"Initialized item embeddings with precomputed values")

            # Resize BERT embeddings to match vocabulary size
            self.transformer_model.resize_token_embeddings(vocab_size)
            print(f"Resized BERT token embeddings to vocab size: {vocab_size}")

            # Copy item embeddings to BERT embeddings
            with torch.no_grad():
                self.transformer_model.embeddings.word_embeddings.weight.copy_(
                    self.item_embeddings.weight
                )
            print(f"Copied item embeddings to BERT embeddings")
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=bert_config['hidden_size'],
                padding_idx=padding_idx
            )
            # Resize BERT embeddings
            self.transformer_model.resize_token_embeddings(vocab_size)
            print(f"Initialized and resized item embeddings to vocab size: {vocab_size}")

        if self.add_head:
            self.head = nn.Linear(
                bert_config['hidden_size'],
                vocab_size,
                bias=True  # Changed to match BERT's output layer
            )
            if self.tie_weights:
                self.head.weight = self.item_embeddings.weight
            print(f"Initialized head with tied weights: {self.tie_weights}")

        # Initialize weights if no precomputed embeddings
        if precomputed_item_embeddings is None:
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
            print(f"Initialized item embeddings with normal distribution")
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()
            print(f"Zeroed padding index: {self.padding_idx}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get embeddings and project if needed
        embeds = self.item_embeddings(input_ids)
        if hasattr(self, 'embedding_projection'):
            embeds = self.embedding_projection(embeds)
            print(f"Projected embeddings in forward pass")

        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)
            # print(f"Applied head in forward pass")

        return outputs

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """Resize token embeddings and ensure weight tying is maintained."""
        # Resize BERT embeddings
        self.transformer_model.resize_token_embeddings(new_num_tokens)
        print(f"Resized BERT token embeddings to new num tokens: {new_num_tokens}")

        if new_num_tokens is None:
            return self.item_embeddings

        # Create new embeddings
        old_embeddings = self.item_embeddings
        self.item_embeddings = nn.Embedding(
            new_num_tokens,
            self.bert_config['hidden_size'],
            padding_idx=self.padding_idx
        )
        
        # Copy weights for existing tokens
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        self.item_embeddings.weight.data[:num_tokens_to_copy] = \
            old_embeddings.weight.data[:num_tokens_to_copy]
        print(f"Copied weights for existing tokens")

        # Update head if weights are tied
        if self.tie_weights:
            self.head.weight = self.item_embeddings.weight
            print(f"Updated head weights to tie with item embeddings")

        return self.item_embeddings