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
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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


class LMDataset(Dataset):
    def __init__(self, df, max_length=128, num_negatives=None, full_negative_sampling=True,
                 user_col='user_id', item_col='item_id', time_col='timestamp'):
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.full_negative_sampling = full_negative_sampling
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

        if num_negatives:
            self.all_items = df[item_col].unique()

    def __len__(self):
        return len(self.data)

    def sample_negatives(self, item_sequence):
        negatives = np.array(list(set(self.all_items) - set(item_sequence)))
        if self.full_negative_sampling:
            negatives = np.random.choice(
                negatives, size=self.num_negatives * (len(item_sequence) - 1), replace=True)
            negatives = negatives.reshape(len(item_sequence) - 1, self.num_negatives)
        else:
            negatives = np.random.choice(negatives, size=self.num_negatives, replace=False)
        return negatives

class MaskedLMPredictionDataset(LMDataset):
    def __init__(self, df, max_length=128, masking_value=1,
                 validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):
        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)
        self.masking_value = masking_value
        self.validation_mode = validation_mode

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]

        if self.validation_mode:
            target = item_sequence[-1]
            input_ids = item_sequence[-self.max_length:-1]
            item_sequence = item_sequence[:-1]
        else:
            input_ids = item_sequence[-self.max_length + 1:]

        input_ids += [self.masking_value]

        if self.validation_mode:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence, 'target': target}
        else:
            return {'input_ids': input_ids, 'user_id': user_id,
                    'full_history': item_sequence}

class PaddingCollateFn:
    def __init__(self, padding_value=0, labels_padding_value=-100):
        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):
        collated_batch = {}

        for key in batch[0].keys():
            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.stack([torch.tensor(example[key]) for example in batch])
                continue

            if key == 'labels':
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value

            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                             padding_value=padding_value)

        if 'input_ids' in collated_batch:
            attention_mask = collated_batch['input_ids'] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch

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
        model_config = checkpoint['config']
        
        # Create prediction dataset using the original implementation
        val_dataset = MaskedLMPredictionDataset(
            val,
            max_length=128,
            masking_value=model_config['vocab_size'] - 2,  # Second to last token for mask
            validation_mode=True
        )
        
        # Create dataloader with the original collate function
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=PaddingCollateFn(
                padding_value=model_config['vocab_size'] - 1  # Last token for padding
            )
        )
        
        # Get item embeddings
        item_embs = load_embeddings(model_name, train, ie)
        
        # Create model using original BERT4Rec class
        model = BERT4Rec(
            vocab_size=model_config['vocab_size'],
            bert_config=model_config,
            precomputed_item_embeddings=item_embs,
            padding_idx=model_config['vocab_size'] - 2
        )

        print(f"Loaded model config hidden_size: {model_config['hidden_size']}")
        print(f"Expected hidden_size: {model.bert_config['hidden_size']}")

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        
        # Generate recommendations using proper batching
        all_metrics_val = []
        if isinstance(k, int):
            k = [k]
            
        for current_k in k:
            user_recommendations = {}
            
            for batch in tqdm(val_loader, desc=f'Generating recommendations for k={current_k}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_ids = batch['user_id'].numpy()
                histories = batch['full_history']
                targets = batch['target'].numpy()
                
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask)
                    print(f"Model output shape: {outputs.shape}")  # Should be (batch_size, seq_len, vocab_size)
                    scores = outputs[:, -1, :-2]  # Last position, exclude mask and pad tokens
                    
                    # Get top-k items
                    _, top_items = torch.topk(scores, k=current_k, dim=-1)
                    top_items = top_items.cpu().numpy()
                    
                    for user_id, history, recommendations in zip(user_ids, histories, top_items):
                        filtered_recs = [
                            item for item in recommendations 
                            if item not in history
                        ][:current_k]
                        user_recommendations[user_id] = filtered_recs
            
            # Calculate metrics
            df = dict_to_pandas(user_recommendations)
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

    else:
        # Original initialization code for new models...
        model_config = {
            'vocab_size': len(train['item_id'].unique()),
            'max_position_embeddings': 200,
            'hidden_size': 256,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'intermediate_size': 1024
        }
        
        item_embs = load_embeddings(model_name, train, ie)
        
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