from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT4Rec(nn.Module):
    """
    BERT4Rec model for sequential recommendation.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique items).
        bert_config (Dict[str, Any]): Configuration dictionary for the BERT model.
        precomputed_item_embeddings (Optional[np.ndarray], optional): Precomputed item embeddings. Defaults to None.
        add_head (bool, optional): Whether to add a prediction head. Defaults to True.
        tie_weights (bool, optional): Whether to tie the weights of the prediction head to the item embeddings. Defaults to True.
        padding_idx (int, optional): Index of the padding token. Defaults to -1.
        init_std (float, optional): Standard deviation for weight initialization. Defaults to 0.02.
    """
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        # Initialize item embeddings
        if precomputed_item_embeddings is not None:
            precomputed_item_embeddings = torch.from_numpy(
                precomputed_item_embeddings.astype(np.float32)
            )
            hidden_size = bert_config['hidden_size']
            projection = nn.Linear(precomputed_item_embeddings.shape[1], hidden_size)
            precomputed_item_embeddings = projection(precomputed_item_embeddings)
            self.item_embeddings = nn.Embedding.from_pretrained(
                precomputed_item_embeddings,
                padding_idx=padding_idx
            )
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=bert_config['hidden_size'],
                padding_idx=padding_idx
            )

        # Initialize BERT model
        self.transformer_model = BertModel(BertConfig(**bert_config))

        # Add prediction head if specified
        if self.add_head:
            self.head = nn.Linear(
                bert_config['hidden_size'],
                vocab_size,
                bias=False
            )
            if self.tie_weights:
                self.head.weight = nn.Parameter(self.item_embeddings.weight)

        # Initialize weights
        if precomputed_item_embeddings is None:
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()

    def freeze_item_embs(self, flag: bool) -> None:
        """
        Freeze or unfreeze item embeddings.

        Args:
            flag (bool): If True, freeze the item embeddings. If False, unfreeze them.
        """
        self.item_embeddings.weight.requires_grad = flag

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input item IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        # Get item embeddings
        embeds = self.item_embeddings(input_ids)
        
        # Pass embeddings through the transformer model
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        # Apply prediction head if specified
        if self.add_head:
            outputs = self.head(outputs)

        return outputs