from sklearn.calibration import LabelEncoder
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import os
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Any, List, Optional
import pandas as pd
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split

class BERT4Rec(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        bert_config: Dict[str, Any],
        model_names: List[str] = None,
        precomputed_item_embeddings: Optional[np.ndarray] = None,
        add_head: bool = True,
        tie_weights: bool = True,
        padding_idx: int = -1,
        init_std: float = 0.02,
        projection_dim: int = 256
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
            # Project embeddings to desired dimension
            projection = nn.Linear(precomputed_item_embeddings.size(1), projection_dim)
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

        self.transformer_model = BertModel(BertConfig(**bert_config))

        if self.add_head:
            self.head = nn.Linear(
                bert_config['hidden_size'],
                vocab_size,
                bias=False
            )
            if self.tie_weights:
                self.head.weight = nn.Parameter(self.item_embeddings.weight)

        if precomputed_item_embeddings is None:
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()

    def freeze_item_embs(self, flag: bool) -> None:
        self.item_embeddings.weight.requires_grad = flag

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.item_embeddings(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs
    

def train_bert4rec(
    train_data: str,
    val_data: str,
    model_save_path: str,
    model_names: List[str],
    batch_size: int = 128,
    num_epochs: int = 200,
    hidden_dim: int = 256,
    lr: float = 0.001
) -> None:
    """Train BERT4Rec model"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BERT4Rec training")
    
    train = pd.read_parquet(train_data)
    val = pd.read_parquet(val_data)

    # Prepare data and get embeddings
    train, val, _, ie = prepare_bert_data(train, val, None, 'val')
    precomputed_embeddings = load_embeddings(model_names, train, ie)


    # Model configuration
    model_config = {
        'vocab_size': len(train['track_id'].unique()),
        'max_position_embeddings': 200,
        'hidden_size': hidden_dim,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': hidden_dim * 4
    }
    
    # Initialize model
    model = BERT4Rec(
        vocab_size=model_config['vocab_size'],
        bert_config=model_config,
        precomputed_item_embeddings=precomputed_embeddings,
        padding_idx=model_config['vocab_size'] - 1
    )
    model.to('cuda')

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'model_names': model_names
    }, model_save_path)

def load_bert4rec(model_path: str) -> BERT4Rec:
    """Load trained BERT4Rec model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = BERT4Rec(
        vocab_size=checkpoint['config']['vocab_size'],
        bert_config=checkpoint['config'],
        padding_idx=checkpoint['config']['vocab_size'] - 1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_embeddings(model_names: List[str], train: pd.DataFrame, ie: LabelEncoder):
    """
    Load and combine multiple embeddings.
    
    Parameters:
        model_names (List[str]): List of embedding model names
        train (pd.DataFrame): Training data
        ie (LabelEncoder): Item label encoder
    
    Returns:
        np.ndarray: Combined embeddings matrix
    """
    all_embeddings = []
    
    for model_name in model_names:
        if 'mfcc' in model_name:
            _, emb_size = safe_split(model_name)
            emb_size = int(emb_size) if emb_size is not None else 104
            item_embs = pd.read_parquet(f'embeddings/{model_name.split("_")[0]}.pqt').reset_index()
            item_embs = item_embs[item_embs.columns[:emb_size]]
        else:
            item_embs = pd.read_parquet(f'embeddings/{model_name}.pqt').reset_index()
        
        item_embs['track_id'] = item_embs['track_id'].apply(lambda x: x.split('.')[0])
        item_embs = item_embs[item_embs.track_id.isin(train.track_id.unique())].reset_index(drop=True)
        item_embs.index = ie.transform(item_embs.track_id).astype('int')
        item_embs = item_embs.drop(['track_id'], axis=1).astype('float32')
        item_embs = item_embs.loc[list(np.sort(train.item_id.unique()))].values
        
        all_embeddings.append(item_embs)
    
    # Concatenate all embeddings
    combined_embeddings = np.concatenate(all_embeddings, axis=1)
    return combined_embeddings


def prepare_bert_data(train, val, test, mode='val'):
    """Prepare data for BERT4Rec model using existing utils"""
    if mode == 'test':
        train = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
        val = test
    del test
    
    train['item_id'] = train['track_id']
    val['item_id'] = val['track_id']

    ue = LabelEncoder()
    ie = LabelEncoder()
    train['user_id'] = ue.fit_transform(train['user_id'])
    train['item_id'] = ie.fit_transform(train['track_id'])
    val['user_id'] = ue.transform(val['user_id'])
    val['item_id'] = ie.transform(val['track_id'])

    user_history = train.groupby('user_id', observed=False)['item_id'].agg(set).to_dict()
    return train, val, user_history, ie

def calc_bert4rec(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    mode: str = 'val',
    suffix: str = "cosine",
    k: int | list = 50
) -> pd.DataFrame:
    """Calculate metrics for BERT4Rec model"""
    run_name = f'{model_name}_{suffix}'
    train, val, user_history, ie = prepare_bert_data(train, val, test, mode)
    
    # Load model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = f'checkpoints/BERT4Rec/{run_name}_best.pt'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = BERT4Rec(
        vocab_size=len(train['item_id'].unique()),
        bert_config=checkpoint['config'],
        padding_idx=len(train['item_id'].unique()) - 1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Get embeddings
    item_embs = model.item_embeddings.weight.detach().cpu().numpy()
    
    all_users = val.user_id.unique()
    if isinstance(k, int):
        k = [k]

    all_metrics_val = []
    
    for current_k in k:
        user_recommendations = {}
        for user_id in tqdm(all_users, desc=f'applying BERT4Rec for {model_name} with k={current_k}'):
            history = user_history.get(user_id, set())
            user_items = torch.LongTensor([list(history)]).to(device)
            with torch.no_grad():
                user_vector = model.item_embeddings(user_items).mean(dim=1).cpu().numpy()[0]
            
            scores = np.dot(item_embs, user_vector)
            recommendations = np.argsort(scores)[::-1]
            filtered_recommendations = [idx for idx in recommendations if idx not in history][:current_k]
            user_recommendations[user_id] = filtered_recommendations
            
        df = dict_to_pandas(user_recommendations)
        
        os.makedirs('metrics', exist_ok=True)
        metrics_val = calc_metrics(val, df, current_k)
        metrics_val = metrics_val.apply(mean_confidence_interval)
        
        if len(k) > 1:
            metrics_val.columns = [f'{col.split("@")[0]}@k' for col in metrics_val.columns]
            metrics_val.index = [f'mean at k={current_k}', f'CI at {current_k=}']
            
        all_metrics_val.append(metrics_val)
    
    if len(k) > 1:
        metrics_val_concat = pd.concat(all_metrics_val, axis=0)
    else:
        metrics_val_concat = all_metrics_val[0]
        
    metrics_val_concat.to_csv(f'metrics/{run_name}_val.csv')
    
    return metrics_val_concat

def bert4rec(
    model_names: str | list,
    suffix: str,
    k: int | list,
    mode: str = 'val'
) -> pd.DataFrame:
    """Main function to evaluate BERT4Rec models"""
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/validation.pqt')
    test = pd.read_parquet('data/test.pqt')
    
    if isinstance(model_names, str):
        return calc_bert4rec(model_names, train, val, test, mode, suffix, k)
    else:
        return pd.concat([
            calc_bert4rec(model_name, train, val, test, mode, suffix, k) 
            for model_name in model_names
        ])