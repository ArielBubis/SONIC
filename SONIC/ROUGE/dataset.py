from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class MaskedLMDataset(Dataset):
    """Dataset for masked language modeling training."""
    def __init__(
        self, 
        df: pd.DataFrame, 
        max_length: int = 128,
        mlm_probability: float = 0.2,
        masking_value: int = 1,
        ignore_value: int = -100,
        force_last_item_masking_prob: float = 0,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        time_col: str = 'timestamp'
    ):
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.masking_value = masking_value
        self.ignore_value = ignore_value
        self.force_last_item_masking_prob = force_last_item_masking_prob
        
        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_sequence = self._get_truncated_sequence(idx)
        input_ids, labels = self._create_masked_sequence(item_sequence)
        return {'input_ids': input_ids, 'labels': labels}

    def _get_truncated_sequence(self, idx):
        """Get sequence truncated to max_length."""
        sequence = self.data[self.user_ids[idx]]
        if len(sequence) > self.max_length:
            sequence = sequence[-self.max_length:]
        return sequence

    def _create_masked_sequence(self, sequence):
        """Create masked sequence for training."""
        input_ids = np.array(sequence)
        mask = np.random.rand(len(sequence)) < self.mlm_probability
        input_ids[mask] = self.masking_value
        
        if self.force_last_item_masking_prob > 0:
            if np.random.rand() < self.force_last_item_masking_prob:
                input_ids[-1] = self.masking_value
                
        labels = np.array(sequence)
        labels[input_ids != self.masking_value] = self.ignore_value
        
        return input_ids, labels

class PredictionDataset(Dataset):
    """Dataset for generating recommendations."""
    def __init__(
        self,
        df: pd.DataFrame,
        max_length: int = 128,
        masking_value: int = 1,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        time_col: str = 'timestamp'
    ):
        self.max_length = max_length
        self.masking_value = masking_value
        
        self.data = df.sort_values(time_col).groupby(user_col)[item_col].agg(list).to_dict()
        self.user_ids = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.data[user_id]
        input_ids = item_sequence[-self.max_length + 1:] + [self.masking_value]
        
        return {
            'input_ids': input_ids,
            'user_id': user_id,
            'full_history': item_sequence
        }