import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Any, Optional, Tuple
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
    init_std: float = 0.02,
    projection_dim: Optional[int] = None,
    projection_strategy: str = 'linear'  # New parameter
):
        """
        Initialize BERT4Rec model.
        
        Args:
            vocab_size: Number of items in vocabulary
            bert_config: Configuration for BERT model
            precomputed_item_embeddings: Pre-trained item embeddings
            add_head: Whether to add output head
            tie_weights: Whether to tie input and output weights
            padding_idx: Index used for padding
            init_std: Standard deviation for weight initialization
            projection_dim: Dimension for projection layer (if needed)
        """
        super().__init__()
        self._validate_inputs(vocab_size, bert_config, precomputed_item_embeddings)
        
        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std
        
        # Set up embedding and projection layers
        self._setup_embeddings(
        precomputed_item_embeddings, 
        projection_dim,
        projection_strategy
    )
        
        # Initialize BERT transformer
        self.transformer_model = BertModel(BertConfig(**bert_config))
        
        # Set up output head if needed
        if self.add_head:
            self._setup_output_head()
        
        # Initialize all weights
        self._initialize_weights()
        
        logging.info(f"Initialized BERT4Rec with vocab size {vocab_size} and hidden size {bert_config['hidden_size']}")

    def _validate_inputs(
        self, 
        vocab_size: int,
        bert_config: Dict[str, Any],
        precomputed_item_embeddings: Optional[np.ndarray]
    ) -> None:
        """Validate input parameters."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
            
        if 'hidden_size' not in bert_config:
            raise ValueError("bert_config must contain 'hidden_size'")
            
        if precomputed_item_embeddings is not None:
            if len(precomputed_item_embeddings.shape) != 2:
                raise ValueError("precomputed_item_embeddings must be 2-dimensional")
            if precomputed_item_embeddings.shape[0] > vocab_size:
                raise ValueError(
                    f"Number of precomputed embeddings ({precomputed_item_embeddings.shape[0]}) "
                    f"cannot exceed vocab_size ({vocab_size})"
                )

    def _setup_embeddings(
    self,
    precomputed_item_embeddings: Optional[np.ndarray],
    projection_dim: Optional[int],
    projection_strategy: str = 'linear'  # New parameter
    ) -> None:
        if precomputed_item_embeddings is not None:
            input_dim = precomputed_item_embeddings.shape[1]
            hidden_dim = self.bert_config['hidden_size']
            
            self.item_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(precomputed_item_embeddings.astype(np.float32)),
                padding_idx=self.padding_idx
            )
            
            if input_dim != hidden_dim:
                if projection_strategy == 'linear':
                    self.projection = self._create_projection_layers(
                        input_dim, hidden_dim, hidden_dim, num_layers=1
                    )
                elif projection_strategy == 'progressive':
                    if projection_dim is None:
                        projection_dim = (input_dim + hidden_dim) // 2
                    self.projection = self._create_projection_layers(
                        input_dim, projection_dim, hidden_dim, num_layers=2
                    )
                elif projection_strategy == 'deep':
                    self.projection = self._create_projection_layers(
                        input_dim, projection_dim, hidden_dim, num_layers=3
                    )
                
                logging.info(
                    f"Created {projection_strategy} projection: {input_dim} -> {hidden_dim}"
                )
            else:
                self.projection = nn.Identity()
                logging.info("No projection needed - dimensions match")
        else:
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.bert_config['hidden_size'],
                padding_idx=self.padding_idx
            )
            self.projection = nn.Identity()

    def _create_projection_layers(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        num_layers: int = 2  # New parameter
    ) -> nn.Sequential:
        """Create projection layers with specified dimensions and depth."""
        if num_layers == 1:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            
        layers = []
        dims = np.linspace(input_dim, hidden_dim, num_layers + 1).astype(int)
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU() if i < num_layers - 1 else nn.Identity()
            ])
        
        return nn.Sequential(*layers)

    def _setup_output_head(self) -> None:
        """Set up the output head layer."""
        self.head = nn.Linear(
            self.bert_config['hidden_size'],
            self.vocab_size,
            bias=False
        )
        
        # Only tie weights if no projection is used
        if self.tie_weights and isinstance(self.projection, nn.Identity):
            self.head.weight = nn.Parameter(self.item_embeddings.weight)
            logging.info("Tied input and output weights")

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        # Initialize embedding weights if not pretrained
        if not hasattr(self, 'item_embeddings_pretrained'):
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
        
        # Set padding embeddings to zero
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()
        
        # Initialize projection layers if they exist
        if not isinstance(self.projection, nn.Identity):
            for layer in self.projection.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(mean=0.0, std=self.init_std)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    def freeze_embeddings(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze item embeddings and projection layers.
        
        Args:
            freeze: Whether to freeze (True) or unfreeze (False) the layers
        """
        self.item_embeddings.weight.requires_grad = not freeze
        if not isinstance(self.projection, nn.Identity):
            for param in self.projection.parameters():
                param.requires_grad = not freeze
        
        status = "Frozen" if freeze else "Unfrozen"
        logging.info(f"{status} embeddings and projection layers")

    def get_embedding_weights(self) -> torch.Tensor:
        """Get the final item embedding weights after projection."""
        with torch.no_grad():
            ids = torch.arange(self.vocab_size).to(next(self.parameters()).device)
            embeddings = self.item_embeddings(ids)
            projected = self.projection(embeddings)
        return projected

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Model outputs
        """
        # Get and project embeddings
        embeds = self.item_embeddings(input_ids)
        embeds = self.projection(embeds)
        
        # Pass through transformer
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state
        
        # Apply output head if it exists
        if self.add_head:
            outputs = self.head(outputs)
            
        return outputs

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for analysis.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (outputs, attention_weights)
        """
        embeds = self.item_embeddings(input_ids)
        embeds = self.projection(embeds)
        
        outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        return outputs.last_hidden_state, outputs.attentions