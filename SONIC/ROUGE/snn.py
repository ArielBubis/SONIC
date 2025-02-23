import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split

# Create data loaders
from SONIC.ROUGE.datasets import InteractionDataset
from SONIC.ROUGE.snn_model import ShallowEmbeddingModel
# Initialize model

# class ShallowEmbeddingModel(nn.Module):
#     """
#     A shallow embedding model for user-item interactions.

#     Args:
#         num_users (int): Number of users.
#         num_items (int): Number of items.
#         emb_dim_in (int): Input embedding dimension.
#         precomputed_item_embeddings (np.ndarray, optional): Precomputed item embeddings. Defaults to None.
#         precomputed_user_embeddings (np.ndarray, optional): Precomputed user embeddings. Defaults to None.
#         emb_dim_out (int, optional): Output embedding dimension. Defaults to 300.
#     """
#     def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, precomputed_user_embeddings=None, emb_dim_out=300):
#         super(ShallowEmbeddingModel, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.emb_dim_in = emb_dim_in

#         # Initialize user embeddings
#         if precomputed_user_embeddings is None:
#             self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in)
#         else:
#             precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings)
#             assert precomputed_user_embeddings.size(1) == emb_dim_in
#             self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings)

#         # Initialize item embeddings
#         if precomputed_item_embeddings is None:
#             self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in)
#         else:
#             precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings)
#             assert precomputed_item_embeddings.size(1) == emb_dim_in
#             self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings)

#         # Define the model architecture
#         self.model = nn.Sequential(
#             nn.Linear(self.emb_dim_in, emb_dim_out),
#             nn.ReLU()
#         )

#         self.model.to(self.device)

#         # Cosine similarity for scoring
#         self.cossim = torch.nn.CosineSimilarity()

#     def freeze_item_embs(self, flag):
#         """
#         Freeze or unfreeze item embeddings.

#         Args:
#             flag (bool): If True, freeze the item embeddings. If False, unfreeze them.
#         """
#         self.item_embeddings.weight.requires_grad = flag

#     def freeze_user_embs(self, flag):
#         """
#         Freeze or unfreeze user embeddings.

#         Args:
#             flag (bool): If True, freeze the user embeddings. If False, unfreeze them.
#         """
#         self.user_embeddings.weight.requires_grad = flag

#     def forward(self, user_indices, item_indices):
#         """
#         Forward pass of the model.

#         Args:
#             user_indices (torch.Tensor): Indices of the users.
#             item_indices (torch.Tensor): Indices of the items.

#         Returns:
#             torch.Tensor: Cosine similarity scores between user and item embeddings.
#         """
#         # Get user and item embeddings
#         user_embeds = self.user_embeddings(user_indices).to(self.device)
#         item_embeds = self.item_embeddings(item_indices).to(self.device)

#         # Pass embeddings through the model
#         user_embeds = self.model(user_embeds)
#         item_embeds = self.model(item_embeds)

#         # Compute cosine similarity scores
#         scores = self.cossim(user_embeds, item_embeds)
#         return scores

#     def extract_embeddings(self, normalize=True):
#         """
#         Extract user and item embeddings.

#         Args:
#             normalize (bool, optional): If True, normalize the embeddings. Defaults to True.

#         Returns:
#             tuple: Normalized user and item embeddings.
#         """
#         user_embeddings = self.user_embeddings.weight.data
#         item_embeddings = self.item_embeddings.weight.data

#         with torch.no_grad():
#             user_embeddings = self.model(user_embeddings.to(self.device)).cpu().numpy()
#             item_embeddings = self.model(item_embeddings.to(self.device)).cpu().numpy()

#         if normalize:
#             user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
#             item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

#         return user_embeddings, item_embeddings


def load_embeddings(model_name, train, ie, use_lyrics=False, use_metadata=False):
    """
    Load item embeddings from a parquet file with optional lyrics and metadata embeddings.

    Args:
        model_name (str): Name of the model.
        train (pd.DataFrame): Training data.
        ie (LabelEncoder): Item encoder.
        use_lyrics (bool, optional): Whether to use lyrics embeddings. Defaults to False.
        use_metadata (bool, optional): Whether to use metadata embeddings. Defaults to False.

    Returns:
        np.ndarray: Loaded item embeddings.
    """
    # Load base audio embeddings
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

    # Merge lyrics embeddings if required
    if use_lyrics:
        lyrics_embs = pd.read_parquet('embeddings/lyrical.pqt').astype('float32').reset_index()
        lyrics_embs['track_id'] = lyrics_embs['track_id'].apply(lambda x: x.split('/')[-1].split('.')[0])
        lyrics_embs = lyrics_embs[lyrics_embs.track_id.isin(train.track_id.unique())] \
            .reset_index(drop=True).set_index('track_id')
        lyrics_embs.columns = [f'lyrical_{col}' for col in lyrics_embs.columns]
        item_embs.columns = [f'{col}@{model_name}' if col != 'track_id' else 'track_id' for col in item_embs.columns]
        item_embs = item_embs.merge(lyrics_embs, on='track_id')

    # Merge metadata embeddings if required
    if use_metadata:
        min_max_scaler = MinMaxScaler()
        metadata = pd.read_parquet('data/id_metadata.pqt')[['danceability', 'energy', 'valence', 'tempo']]
        metadata_normed = pd.DataFrame(min_max_scaler.fit_transform(metadata), columns=metadata.columns)
        metadata_normed.columns = [f'metadata_{col}' for col in metadata.columns]
        metadata_normed['track_id'] = metadata.index
        item_embs = item_embs.merge(metadata_normed, on='track_id')

    item_embs = item_embs.drop(['track_id'], axis=1).astype('float32')
    item_embs = item_embs.loc[list(np.sort(train.item_id.unique()))].values

    return item_embs


def prepare_data(train, val, test, mode='val'):
    """
    Prepare data for training and validation.

    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        mode (str, optional): Mode of operation ('val' or 'test'). Defaults to 'val'.

    Returns:
        tuple: Prepared training data, validation data, user history, and item encoder.
    """
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


def calc_snn(model_name, train, val, test, mode='val', suffix='cosine', k=50, emb_dim_out=300, 
           use_lyrics=False, use_metadata=False, num_epochs=100, batch_size=10000, neg_samples=20,
           patience_threshold=16, l2=0, use_confidence=False):
    """
    Calculate recommendations using a shallow neural network model with training.

    Args:
        model_name (str): Name of the model
        train (pd.DataFrame): Training data
        val (pd.DataFrame): Validation data
        test (pd.DataFrame): Test data
        mode (str, optional): Mode of operation ('val' or 'test'). Defaults to 'val'
        suffix (str, optional): Suffix for the run name. Defaults to 'cosine'
        k (int or list, optional): Number of recommendations to generate. Defaults to 50
        emb_dim_out (int, optional): Output embedding dimension. Defaults to 300
        use_lyrics (bool, optional): Whether to use lyrics embeddings. Defaults to False
        use_metadata (bool, optional): Whether to use metadata embeddings. Defaults to False
        num_epochs (int, optional): Number of epochs to train. Defaults to 100
        batch_size (int, optional): Batch size for training. Defaults to 10000
        neg_samples (int, optional): Number of negative samples per positive sample. Defaults to 20
        patience_threshold (int, optional): Number of epochs to wait before early stopping. Defaults to 16
        l2 (float, optional): L2 regularization weight. Defaults to 0
        use_confidence (bool, optional): Whether to use confidence scores. Defaults to False

    Returns:
        pd.DataFrame: Validation metrics
    """
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import os
    from datetime import datetime

    lyrical = "" if not use_lyrics else "+lyrics"
    metadata = "" if not use_metadata else "+metadata"
    current_time = datetime.now().strftime("%b%d_%H:%M")
    run_name = f'{model_name}{lyrical}{metadata}_{suffix}_{current_time}'
    
    # Prepare data
    train, val, user_history, ie = prepare_data(train, val, test, mode)
    item_embs = load_embeddings(model_name, train, ie, use_lyrics, use_metadata)

    num_users = train['user_id'].nunique()
    num_items = train['item_id'].nunique()
    emb_dim_in = item_embs.shape[1]
    model = ShallowEmbeddingModel(num_users, num_items, emb_dim_in, 
                                 precomputed_item_embeddings=item_embs, 
                                 emb_dim_out=emb_dim_out)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = InteractionDataset(train, neg_samples=neg_samples)
    val_dataset = InteractionDataset(val, neg_samples=neg_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up tensorboard and checkpoints
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)

    # Train the model
    best_val_loss = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        neg_samples=neg_samples,
        patience_threshold=patience_threshold,
        l2=l2,
        use_confidence=use_confidence,
        device=device,
        writer=writer,
        checkpoint_dir=f'checkpoints/{model_name}',
        run_name=run_name
    )

    # Load best model checkpoint
    checkpoint = torch.load(f'checkpoints/{model_name}/{run_name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract embeddings and generate recommendations
    user_embs, item_embs = model.extract_embeddings(normalize=(suffix == 'cosine'))
    
    all_users = val.user_id.unique()
    
    if isinstance(k, int):
        k = [k]
    max_k = max(k)

    # Generate recommendations
    user_recommendations_maxk = {}
    for user_id in tqdm(all_users, desc=f'Generating recommendations for {run_name}'):
        history = user_history.get(user_id, set())
        user_vector = user_embs[user_id]
        scores = np.dot(item_embs, user_vector)
        recommendations = np.argsort(scores)[::-1]
        filtered_recommendations = [idx for idx in recommendations if idx not in history][:max_k]
        user_recommendations_maxk[user_id] = filtered_recommendations

    # Calculate metrics for different k values
    all_metrics_val = []
    for current_k in k:
        user_recommendations = {
            user_id: recs[:current_k] 
            for user_id, recs in user_recommendations_maxk.items()
        }
        
        df = dict_to_pandas(user_recommendations)
        
        os.makedirs('metrics', exist_ok=True)
        metrics_val = calc_metrics(val, df, current_k)
        metrics_val = metrics_val.apply(mean_confidence_interval)
        all_metrics_val.append(metrics_val)
        
        if len(k) > 1:
            metrics_val.columns = [f'{col.split("@")[0]}@k' for col in metrics_val.columns]
            metrics_val.index = [f'mean at k={current_k}', f'CI at k={current_k}']

    # Combine and save metrics
    if len(k) > 1:
        metrics_val_concat = pd.concat(all_metrics_val, axis=0)
    else:
        metrics_val_concat = all_metrics_val[0]

    metrics_val_concat.to_csv(f'metrics/snn_{run_name}_val.csv')
    
    # Save embeddings
    os.makedirs('model_embeddings', exist_ok=True)
    np.save(f'model_embeddings/{run_name}_users.npy', user_embs)
    np.save(f'model_embeddings/{run_name}_items.npy', item_embs)

    writer.close()
    return metrics_val_concat

def snn(model_names, suffix, k, mode='val', emb_dim_out=300, use_lyrics=False, use_metadata=False,
        num_epochs=100, batch_size=10000, neg_samples=20, patience_threshold=16, l2=0, use_confidence=False):
    """
    Run the shallow neural network model for multiple model names.

    Args:
        model_names (str or list): Model name(s)
        suffix (str): Suffix for the run name
        k (int or list): Number of recommendations to generate
        mode (str, optional): Mode of operation ('val' or 'test'). Defaults to 'val'
        emb_dim_out (int, optional): Output embedding dimension. Defaults to 300
        use_lyrics (bool, optional): Whether to use lyrics embeddings. Defaults to False
        use_metadata (bool, optional): Whether to use metadata embeddings. Defaults to False
        num_epochs (int, optional): Number of epochs to train. Defaults to 100
        batch_size (int, optional): Batch size for training. Defaults to 10000
        neg_samples (int, optional): Number of negative samples per positive sample. Defaults to 20
        patience_threshold (int, optional): Number of epochs to wait before early stopping. Defaults to 16
        l2 (float, optional): L2 regularization weight. Defaults to 0
        use_confidence (bool, optional): Whether to use confidence scores. Defaults to False

    Returns:
        pd.DataFrame: Combined validation metrics for all models
    """
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/validation.pqt')
    test = pd.read_parquet('data/test.pqt')
    
    if isinstance(model_names, str):
        return calc_snn(
            model_names, train, val, test, mode, suffix, k, emb_dim_out, 
            use_lyrics, use_metadata, num_epochs, batch_size, neg_samples,
            patience_threshold, l2, use_confidence
        )
    else:
        return pd.concat([
            calc_snn(
                model_name, train, val, test, mode, suffix, k, emb_dim_out,
                use_lyrics, use_metadata, num_epochs, batch_size, neg_samples,
                patience_threshold, l2, use_confidence
            ) 
            for model_name in model_names
        ])