import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional
import logging
from typing import Tuple, Dict, Set, Union, List
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from SONIC.CREAM.sonic_utils import safe_split

class LMDataset(Dataset):
    """
    Language Modeling Dataset for user-item interactions.

    Args:
        df (pd.DataFrame): DataFrame containing user-item interactions.
        max_length (int): Maximum sequence length.
        num_negatives (int, optional): Number of negative samples. Defaults to None.
        full_negative_sampling (bool, optional): Whether to use full negative sampling. Defaults to True.
        user_col (str, optional): Column name for users. Defaults to 'user_id'.
        item_col (str, optional): Column name for items. Defaults to 'item_id'.
        time_col (str, optional): Column name for timestamps. Defaults to 'timestamp'.
    """
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

class MaskedLMDataset(LMDataset):
    """
    Sample negative items for a given item sequence.

    Args:
        item_sequence (list): List of item IDs in the sequence.

    Returns:
        np.ndarray: Array of negative samples.
    """

    def __init__(self, df, max_length=128,
                 num_negatives=None, full_negative_sampling=True,
                 mlm_probability=0.2,
                 masking_value=1, ignore_value=-100,
                 force_last_item_masking_prob=0,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length, num_negatives, full_negative_sampling,
                         user_col, item_col, time_col)

        self.mlm_probability = mlm_probability
        self.masking_value = masking_value
        self.ignore_value = ignore_value
        self.force_last_item_masking_prob = force_last_item_masking_prob

    def __getitem__(self, idx):

        item_sequence = self.data[self.user_ids[idx]]

        if len(item_sequence) > self.max_length:
            item_sequence = item_sequence[-self.max_length:]

        input_ids = np.array(item_sequence)
        mask = np.random.rand(len(item_sequence)) < self.mlm_probability
        input_ids[mask] = self.masking_value
        if self.force_last_item_masking_prob > 0:
            if np.random.rand() < self.force_last_item_masking_prob:
                input_ids[-1] = self.masking_value

        labels = np.array(item_sequence)
        labels[input_ids != self.masking_value] = self.ignore_value

        if self.num_negatives:
            negatives = self.sample_negatives(item_sequence)
            return {'input_ids': input_ids, 'labels': labels, 'negatives': negatives}

        return {'input_ids': input_ids, 'labels': labels}


class MaskedLMPredictionDataset(LMDataset):
    """
    Masked Language Modeling Dataset for user-item interactions.

    Args:
        df (pd.DataFrame): DataFrame containing user-item interactions.
        max_length (int): Maximum sequence length.
        num_negatives (int, optional): Number of negative samples. Defaults to None.
        full_negative_sampling (bool, optional): Whether to use full negative sampling. Defaults to True.
        mlm_probability (float, optional): Probability of masking an item. Defaults to 0.2.
        masking_value (int, optional): Value used for masking. Defaults to 1.
        ignore_value (int, optional): Value used to ignore labels. Defaults to -100.
        force_last_item_masking_prob (float, optional): Probability of forcing the last item to be masked. Defaults to 0.
        user_col (str, optional): Column name for users. Defaults to 'user_id'.
        item_col (str, optional): Column name for items. Defaults to 'item_id'.
        time_col (str, optional): Column name for timestamps. Defaults to 'timestamp'.
    """

    def __init__(self, df, max_length=128, masking_value=1,
                 validation_mode=False,
                 user_col='user_id', item_col='item_id',
                 time_col='timestamp'):

        super().__init__(df, max_length=max_length, num_negatives=None,
                         user_col=user_col, item_col=item_col, time_col=time_col)

        self.masking_value = masking_value
        self.validation_mode = validation_mode

    def __getitem__(self, idx):
        """
        Get a single data point for masked language modeling.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: Dictionary containing input IDs, labels, and optionally negatives.
        """

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
    """
    Collate function for padding sequences in a batch.

    Args:
        padding_value (int, optional): Value used for padding. Defaults to 0.
        labels_padding_value (int, optional): Value used for padding labels. Defaults to -100.
    """

    def __init__(self, padding_value=0, labels_padding_value=-100):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): List of data points.

        Returns:
            dict: Dictionary containing collated batch data.
        """

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
    

logger = logging.getLogger(__name__)

def prepare_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    mode: str = 'val'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Set[int]], LabelEncoder]:
    """
    Prepare data for training.

    Args:
        train (pd.DataFrame): Training DataFrame.
        val (pd.DataFrame): Validation DataFrame.
        test (pd.DataFrame): Test DataFrame.
        mode (str): Mode of operation ('val' or 'test').

    Returns:
        Tuple: Processed train data, validation data, user history dict, and item encoder.
    """

    if mode not in ['val', 'test']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'val' or 'test'")

    if mode == 'test':
        train = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
        val = test

    train['item_id'] = train['track_id']
    val['item_id'] = val['track_id']

    ue = LabelEncoder()
    ie = LabelEncoder()
    
    try:
        train['user_id'] = ue.fit_transform(train['user_id'])
        train['item_id'] = ie.fit_transform(train['track_id'])
        val['user_id'] = ue.transform(val['user_id'])
        val['item_id'] = ie.transform(val['track_id'])
    except ValueError as e:
        logger.error("Error encoding user/item IDs")
        raise e

    user_history = train.groupby('user_id', observed=False)['item_id'].agg(set).to_dict()
    return train, val, user_history, ie

def load_embeddings(
    model_name: str,
    train_data: pd.DataFrame,
    item_encoder: LabelEncoder,
    base_path: str = '/content/music4all/embeddings',
    add_special_tokens: bool = True,
    special_token_std: float = 0.02
) -> np.ndarray:
    """
    Load and process embeddings.

    Args:
        model_name (str): Name of the embedding model.
        train_data (pd.DataFrame): Training DataFrame.
        item_encoder (LabelEncoder): LabelEncoder for items.
        base_path (str, optional): Base path for embedding files. Defaults to '/content/music4all/embeddings'.
        add_special_tokens (bool, optional): Whether to add special token embeddings. Defaults to True.
        special_token_std (float, optional): Standard deviation for special token embeddings. Defaults to 0.02.

    Returns:
        np.ndarray: Processed embeddings matrix.
    """
    try:
        base_name, emb_size = safe_split(model_name)
        file_path = Path(base_path)
        
        # Handle different embedding types
        if 'mfcc' in model_name.lower():
            emb_size = int(emb_size) if emb_size is not None else 104
            item_embs = pd.read_parquet(file_path / f"{base_name}.pqt").reset_index()
            item_embs = item_embs[item_embs.columns[:emb_size]]
        else:
            item_embs = pd.read_parquet(file_path / f"{model_name}.pqt").reset_index()

        # Process embeddings
        item_embs['track_id'] = item_embs['track_id'].apply(lambda x: x.split('.')[0])
        item_embs = item_embs[item_embs.track_id.isin(train_data.track_id.unique())].reset_index(drop=True)
        item_embs.index = item_encoder.transform(item_embs.track_id).astype('int')
        item_embs = item_embs.drop(['track_id'], axis=1).astype('float32')
        item_embs = item_embs.loc[list(np.sort(train_data.item_id.unique()))].values

        if add_special_tokens:
            special_embs = np.random.normal(0.0, special_token_std, size=(2, item_embs.shape[1]))
            item_embs = np.concatenate([item_embs, special_embs], axis=0)

        return item_embs
        
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise RuntimeError(f"Failed to load embeddings: {str(e)}")

def validate_inputs(
    model_names: Union[str, List[str]],
    suffix: str,
    k: Union[int, List[int]],
    mode: str,
    max_seq_len: int
) -> None:
    """
    Validate input parameters.

    Args:
        model_names (Union[str, List[str]]): Model name(s).
        suffix (str): Suffix for the run name.
        k (Union[int, List[int]]): Number of recommendations to generate.
        mode (str): Mode of operation ('val' or 'test').
        max_seq_len (int): Maximum sequence length.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    if mode not in ['val', 'test']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'val' or 'test'")
    
    if max_seq_len <= 0:
        raise ValueError(f"Invalid max_seq_len: {max_seq_len}. Must be positive")
    
    if isinstance(k, int) and k <= 0:
        raise ValueError(f"Invalid k: {k}. Must be positive")
    elif isinstance(k, list) and (len(k) == 0 or any(ki <= 0 for ki in k)):
        raise ValueError("Invalid k values. All must be positive")
    

class InteractionDataset(Dataset):
    def __init__(self, df, neg_samples=20, device=None):
        """One positive vs 20 negatives"""
        self.df = df
        self.neg_samples = neg_samples

        self.user_history = df.groupby('user_id')['item_id'].agg(list).to_dict()
        all_items = set(df.item_id.unique())
        self.user_negatives = {user: list(all_items - set(self.user_history[user])) for user in self.user_history}
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        user = torch.tensor(sample.user_id, dtype=torch.long, device=self.device)
        positive_item = torch.tensor(sample.item_id, dtype=torch.long, device=self.device)
        negative_items = self._sample_negatives(sample.user_id)
        return user, positive_item, negative_items

    def _sample_negatives(self, user):
        negatives = self.user_negatives[user]
        res = np.random.choice(negatives, self.neg_samples)
        return torch.tensor(res, dtype=torch.long, device=self.device)

class InteractionDatasetItems(Dataset):
    """All positive users and same number of negative users"""
    def __init__(self, df, neg_samples=20):
        self.df = df
        self.neg_samples = neg_samples
        #
        # self.item_history = df.groupby('item_id')['user_id'].agg(list).to_dict()
        # all_users = set(df.user_id.unique())
        # self.item_negatives = {item: list(all_users - set(self.item_history[item])) for item in self.item_history}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        item = torch.tensor(sample.item_id, dtype=torch.long)
        positive_user = torch.tensor(sample.user_id, dtype=torch.long)
        # negative_users = self._sample_negatives(sample.item_id)
        confidence = torch.tensor(self._confidence(sample['count']), dtype=torch.long)
        # return item, positive_user, negative_users, confidence
        return item, positive_user, confidence

    def _sample_negatives(self, item):
        negatives = self.item_negatives[item]
        res = np.random.choice(negatives, self.neg_samples)
        return torch.tensor(res, dtype=torch.long)

    def _confidence(self, x):
        return x