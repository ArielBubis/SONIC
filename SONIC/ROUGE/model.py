import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Any, Optional
import numpy as np
import logging

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
        """Initialize BERT4Rec model with simpler, more robust architecture."""
        super().__init__()
        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        # Initialize BERT model first to ensure proper setup
        self.transformer_model = BertModel(BertConfig(**bert_config))
        logging.info(f"Initialized transformer with config: {bert_config}")

        # Initialize embeddings
        if precomputed_item_embeddings is not None:
            # Convert and prepare embeddings
            precomputed_item_embeddings = torch.from_numpy(
                precomputed_item_embeddings.astype(np.float32)
            )
            
            # Handle dimension mismatch if needed
            if precomputed_item_embeddings.shape[1] != bert_config['hidden_size']:
                logging.info(f"Adjusting embedding dimensions from {precomputed_item_embeddings.shape[1]} to {bert_config['hidden_size']}")
                self.embedding_projection = nn.Linear(
                    precomputed_item_embeddings.shape[1],
                    bert_config['hidden_size']
                )
                with torch.no_grad():
                    precomputed_item_embeddings = self.embedding_projection(precomputed_item_embeddings)
            
            # Initialize embedding layer
            self.item_embeddings = nn.Embedding.from_pretrained(
                precomputed_item_embeddings,
                padding_idx=padding_idx,
                freeze=False  # Allow fine-tuning
            )
            logging.info(f"Initialized embeddings from precomputed values: {precomputed_item_embeddings.shape}")
        else:
            # Initialize from scratch if no precomputed embeddings
            self.item_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=bert_config['hidden_size'],
                padding_idx=padding_idx
            )
            logging.info(f"Initialized embeddings from scratch: {vocab_size} x {bert_config['hidden_size']}")

        # Ensure transformer embeddings match
        self.transformer_model.resize_token_embeddings(vocab_size)
        
        # Add output head if needed
        if self.add_head:
            self.head = nn.Linear(
                bert_config['hidden_size'],
                vocab_size,
                bias=True  # Using bias for better expressivity
            )
            if self.tie_weights:
                # Tie weights if using same dimensions
                if not hasattr(self, 'embedding_projection'):
                    self.head.weight = self.item_embeddings.weight
                    logging.info("Tied embedding and output weights")
                else:
                    logging.info("Skipped weight tying due to dimension projection")

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        if not hasattr(self, 'item_embeddings_pretrained'):
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
            logging.info(f"Initialized embedding weights with std={self.init_std}")
        
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()
            logging.info(f"Zeroed padding token embeddings at index {self.padding_idx}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass incorporating embedding projection if needed."""
        # Get embeddings
        embeds = self.item_embeddings(input_ids)
        
        # Project if dimensions were adjusted
        if hasattr(self, 'embedding_projection'):
            embeds = self.embedding_projection(embeds)

        # Pass through transformer
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        # Apply output head if present
        if self.add_head:
            outputs = self.head(outputs)

        return outputs

    def freeze_embeddings(self, freeze: bool = True):
        """Freeze or unfreeze embedding layers."""
        for param in self.item_embeddings.parameters():
            param.requires_grad = not freeze
        if hasattr(self, 'embedding_projection'):
            for param in self.embedding_projection.parameters():
                param.requires_grad = not freeze
        status = "Frozen" if freeze else "Unfrozen"
        logging.info(f"{status} embedding layers")

    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Get attention weights for analysis."""
        embeds = self.item_embeddings(input_ids)
        if hasattr(self, 'embedding_projection'):
            embeds = self.embedding_projection(embeds)
            
        outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return outputs.attentions