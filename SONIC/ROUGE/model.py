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

        if precomputed_item_embeddings is not None:
            precomputed_item_embeddings = torch.from_numpy(
                precomputed_item_embeddings.astype(np.float32)
            )
            input_dim = precomputed_item_embeddings.size(1)
            # Project to hidden_size from config
            self.embedding_projection = nn.Linear(input_dim, bert_config['hidden_size'])
            
            # Create full embedding matrix with zeros
            full_embeddings = torch.zeros(vocab_size, input_dim)
            full_embeddings[:len(precomputed_item_embeddings)] = precomputed_item_embeddings
            
            self.item_embeddings = nn.Embedding.from_pretrained(
                full_embeddings,
                padding_idx=padding_idx
            )
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=bert_config['hidden_size'],
                padding_idx=padding_idx
            )
            self.embedding_projection = None

        self.transformer_model = BertModel(BertConfig(**bert_config))

        if self.add_head:
            self.head = nn.Linear(
                bert_config['hidden_size'],
                vocab_size,
                bias=False
            )
            if self.tie_weights:
                self.head.weight = nn.Parameter(self.item_embeddings.weight)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        if not hasattr(self, 'item_embeddings_pretrained'):
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()

    def freeze_item_embs(self, flag: bool) -> None:
        """Freeze or unfreeze item embeddings."""
        self.item_embeddings.weight.requires_grad = flag

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        embeds = self.item_embeddings(input_ids)
        if self.embedding_projection is not None:
            embeds = self.embedding_projection(embeds)
            
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs