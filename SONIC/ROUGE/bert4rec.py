from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import os
from .model import BERT4Rec
from .train import ModelTrainer, TrainingConfig
from sklearn.preprocessing import LabelEncoder
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split


def load_embeddings(model_name, train, ie):
    if 'mfcc' in model_name:
        _, emb_size = safe_split(model_name)
        emb_size = int(emb_size) if emb_size is not None else 104
        item_embs = pd.read_parquet(f'/content/music4all/embeddings/{model_name.split("_")[0]}.pqt').reset_index()
        item_embs = item_embs[item_embs.columns[:emb_size]]
    else:
        item_embs = pd.read_parquet(f'/content/music4all/embeddings/{model_name}.pqt').reset_index()

    item_embs['track_id'] = item_embs['track_id'].apply(lambda x: x.split('.')[0])
    item_embs = item_embs[item_embs.track_id.isin(train.track_id.unique())].reset_index(drop=True)
    item_embs.index = ie.transform(item_embs.track_id).astype('int')
    item_embs = item_embs.drop(['track_id'], axis=1).astype('float32')
    item_embs = item_embs.loc[list(np.sort(train.item_id.unique()))].values

    # Add special embeddings
    return item_embs

def prepare_data(train, val, test, mode='val'):
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
    k: int | list = 50,
    pretrained_path: Optional[str] = None
) -> pd.DataFrame:
    """Calculate metrics for BERT4Rec model"""
    run_name = f'{model_name}_{suffix}'
    train, val, user_history, ie = prepare_data(train, val, test, mode)
    
    # Load model checkpoint or create new model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Checkpoint not found at {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Create model with projection layer matching checkpoint
        model_config = checkpoint['config']
        
        # Get item embeddings to determine input dimension
        item_embs = load_embeddings(model_name, train, ie)
        input_dim = item_embs.shape[1]
        
        model = BERT4Rec(
            vocab_size=model_config['vocab_size'],
            bert_config=model_config,
            precomputed_item_embeddings=item_embs,
            padding_idx=model_config['vocab_size'] - 1,
            projection_dim=model_config['hidden_size']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Get embeddings for new model
        item_embs = load_embeddings(model_name, train, ie)
        
        # Initialize new model
        model_config = {
            'vocab_size': len(train['item_id'].unique()),
            'max_position_embeddings': 200,
            'hidden_size': 256,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'intermediate_size': 1024
        }
        
        model = BERT4Rec(
            vocab_size=model_config['vocab_size'],
            bert_config=model_config,
            precomputed_item_embeddings=item_embs,
            padding_idx=model_config['vocab_size'] - 1
        )
    
    model.to(device)
    
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
            metrics_val.index = [f'mean at k={current_k}', f'CI at k={current_k}']
            
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
    mode: str = 'val',
    pretrained_path: Optional[str] = None
) -> pd.DataFrame:
    """Main function to evaluate BERT4Rec models"""
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/validation.pqt')
    test = pd.read_parquet('data/test.pqt')
    
    if isinstance(model_names, str):
        return calc_bert4rec(
            model_names, train, val, test, mode, suffix, k,
            pretrained_path=pretrained_path
        )
    else:
        return pd.concat([
            calc_bert4rec(
                model_name, train, val, test, mode, suffix, k,
                pretrained_path=pretrained_path
            ) 
            for model_name in model_names
        ])