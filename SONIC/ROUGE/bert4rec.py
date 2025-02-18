from typing import Dict, Optional
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import os
from .model import BERT4Rec
from sklearn.preprocessing import LabelEncoder
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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
    """Calculate metrics for BERT4Rec model with improved embedding handling."""
    run_name = f'{model_name}_{suffix}'
    os.makedirs('metrics', exist_ok=True)
    train, val, user_history, ie = prepare_data(train, val, test, mode)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if pretrained_path:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Checkpoint not found at {pretrained_path}")
        
        # Load checkpoint and model configuration
        checkpoint = torch.load(pretrained_path, map_location=device)
        model_config = checkpoint['config']

        # Calculate correct vocabulary sizes
        checkpoint_hidden_size = model_config['hidden_size']
        base_vocab_size = len(train['item_id'].unique())  # Number of actual items
        num_special_tokens = 2  # Mask and padding tokens
        total_vocab_size = base_vocab_size + num_special_tokens

        # Load and process embeddings
        item_embs = load_embeddings(model_name, val, ie)
        input_dim = item_embs.shape[1]

        print(f"Dimensions:")
        print(f"- Input embedding dim: {input_dim}")
        print(f"- Checkpoint hidden size: {checkpoint_hidden_size}")
        print(f"- Vocabulary size: {total_vocab_size}")
        if input_dim != checkpoint_hidden_size:
            print(f"Projecting embeddings from {input_dim} to {checkpoint_hidden_size}")
            projection = nn.Linear(input_dim, checkpoint_hidden_size)
            item_embs_tensor = torch.from_numpy(item_embs.astype(np.float32))
            with torch.no_grad():
                item_embs = projection(item_embs_tensor).numpy()

        # Pad embeddings if needed
        if item_embs.shape[0] < base_vocab_size:
            padding = np.zeros((
                base_vocab_size - item_embs.shape[0],
                checkpoint_hidden_size  # Use checkpoint dimension
            ))
            item_embs = np.vstack([item_embs, padding])

        # Add special token embeddings
        special_token_embeddings = np.zeros((num_special_tokens, checkpoint_hidden_size))
        item_embs = np.vstack([item_embs, special_token_embeddings])

        # Update model config
        model_config['vocab_size'] = total_vocab_size
        model_config['hidden_size'] = checkpoint_hidden_size

        # checkpoint_vocab_size = model_config['vocab_size']
        # Ensure padding_idx is within valid range
        # Set token indices
        mask_token_idx = total_vocab_size - 2
        padding_idx = total_vocab_size - 1

        print(f"Vocabulary details:")
        print(f"- Base vocab size: {base_vocab_size}")
        print(f"- Total vocab size: {total_vocab_size}")
        print(f"- Mask token index: {mask_token_idx}")
        print(f"- Padding token index: {padding_idx}")

        # Create datasets with proper masking and padding values
        val_dataset = MaskedLMPredictionDataset(
            val,
            max_length=128,
            masking_value=mask_token_idx,
            validation_mode=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=PaddingCollateFn(
                padding_value=padding_idx
            )
        )
        item_embs = load_embeddings(model_name, val, ie)
        if item_embs.shape[0] < base_vocab_size:
            padding = np.zeros((
                base_vocab_size - item_embs.shape[0],
                item_embs.shape[1]
            ))
            item_embs = np.vstack([item_embs, padding])


        # # # Adjust embedding dimensions if needed
        # if input_dim != model_config['hidden_size']:
        #     print(f"Projecting embeddings from {input_dim} to {model_config['hidden_size']}")
        #     item_embs = item_embs.astype(np.float32)
            
        #     # Handle padding for vocabulary size
        #     if item_embs.shape[0] < checkpoint_vocab_size - 3:
        #         padding = np.zeros((
        #             checkpoint_vocab_size - 3 - item_embs.shape[0],
        #             item_embs.shape[1]
        #         ))
        #         item_embs = np.vstack([item_embs, padding])
        #         print(f"Padded embeddings to match vocabulary size: {item_embs.shape}")

        # Add embeddings for special tokens
        special_token_embeddings = np.zeros((num_special_tokens, item_embs.shape[1]))
        item_embs = np.vstack([item_embs, special_token_embeddings])

        # Update model config
        model_config['vocab_size'] = total_vocab_size
        # Initialize model with processed embeddings
        model = BERT4Rec(
            vocab_size=total_vocab_size,
            bert_config=model_config,
            precomputed_item_embeddings=item_embs,
            padding_idx=padding_idx  # Use corrected padding_idx
        )

        
        # # Log configuration details
        # print(f"Model Configuration:")
        # print(f"- Vocabulary Size: {checkpoint_vocab_size}")
        # print(f"- Hidden Size: {model_config['hidden_size']}")
        # print(f"- Input Embedding Dimension: {input_dim}")
        # print(f"- Number of Items: {len(train['item_id'].unique())}")
        
        # Load checkpoint weights
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            print("Warning: Some weights were not loaded:")
            print(f"- Missing keys: {incompatible_keys.missing_keys}")
            print(f"- Unexpected keys: {incompatible_keys.unexpected_keys}")
        
        model.to(device)
        model.eval()  # Ensure evaluation mode
        
        # Generate recommendations and calculate metrics
        all_metrics_val = []
        k_values = [k] if isinstance(k, int) else k
        
        for current_k in k_values:
            # Generate recommendations
            user_recommendations = generate_recommendations(
                model=model,
                pred_loader=val_loader,
                user_history=user_history,
                device=device,
                k=current_k
            )
            
            # Calculate metrics
            df = dict_to_pandas(user_recommendations)
            metrics_val = calc_metrics(val, df, current_k)
            metrics_val = metrics_val.apply(mean_confidence_interval)
            
            # Format metric names
            if len(k_values) > 1:
                metrics_val.columns = [f'{col.split("@")[0]}@k' for col in metrics_val.columns]
                metrics_val.index = [f'mean at k={current_k}', f'CI at k={current_k}']
            
            all_metrics_val.append(metrics_val)
        
        # Combine metrics for different k values
        metrics_val_concat = (
            pd.concat(all_metrics_val, axis=0) if len(k_values) > 1 
            else all_metrics_val[0]
        )
        
        # Save results
        metrics_val_concat.to_csv(f'metrics/{run_name}_val.csv')
        print(f"Saved metrics to metrics/{run_name}_val.csv")
        
        return metrics_val_concat
    else:
        raise NotImplementedError("Pretrained model path is required for evaluation.")
    

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
    

def generate_recommendations(
    model: nn.Module,
    pred_loader: DataLoader,
    user_history: Dict[int, list],
    device: torch.device,
    k: int = 100
) -> Dict[int, list]:
    """Generate recommendations with improved batch processing."""
    model.eval()
    user_recommendations = {}
    
    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Generating recommendations"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_ids = batch['user_id']
            
            # Get model outputs
            outputs = model(input_ids, attention_mask)
            
            # Get last item predictions
            seq_lengths = attention_mask.sum(dim=1).long()
            last_item_logits = torch.stack([
                outputs[i, seq_lengths[i] - 1, :]
                for i in range(len(seq_lengths))
            ])
            
            # Remove special tokens from predictions
            last_item_logits = last_item_logits[:, :-2]
            
            # Get top-k predictions efficiently
            preds = torch.topk(last_item_logits, k=k, dim=-1).indices
            preds = preds.cpu().numpy()
            
            # Filter recommendations
            for user_id, item_ids in zip(user_ids, preds):
                user_id = user_id.item()
                history = user_history.get(user_id, set())
                recs = [
                    item_id for item_id in item_ids 
                    if item_id not in history
                ][:k]
                user_recommendations[user_id] = recs
    
    return user_recommendations