import os
from pdb import run
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import faiss
from sklearn.preprocessing import LabelEncoder
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split

def prepare_data(train, val, test, mode):
    
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


def prepare_index(train, model_name, suffix, ie, use_lyrics):
    lyrical = "" if not use_lyrics else "+lyrics"
    run_name = f'{model_name}{lyrical}_{suffix}'
    tqdm.pandas(desc=f'computing user embeddings for {run_name}')
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

    if use_lyrics:
        lyrics_embs = pd.read_parquet('embeddings/lyrical.pqt').astype('float32').reset_index()
        lyrics_embs['item_id'] = lyrics_embs['track_id'].apply(lambda x: x.split('/')[-1].split('.')[0])
        item_embs = item_embs.merge(lyrics_embs, on='track_id', how='left')

    item_embs = item_embs.drop(['track_id'], axis=1).astype('float32')
    item_embs = item_embs.loc[list(np.sort(train.item_id.unique()))].values
    user_embs = np.stack(train.groupby('user_id')['item_id'].progress_apply(lambda items: item_embs[items].mean(axis=0)).values)

    if suffix == 'cosine':
        user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
        item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(item_embs.shape[1])
    for item_embs_chunk in np.array_split(item_embs, 100):
        index.add(item_embs_chunk)
    # index.add(item_embs)

    return user_embs, index

def calc_knn(model_name, train, val, test, mode='val', suffix="cosine", k:int|list=50, use_lyrics=False):
    lyrical = "" if not use_lyrics else "+lyrics"
    run_name = f'{model_name}{lyrical}_{suffix}'
    train, val, user_history, ie = prepare_data(train, val, test, mode)

    user_embs, index = prepare_index(train, model_name, suffix, ie, use_lyrics)

    all_users = val.user_id.unique()

    if isinstance(k, int):
        k = [k]  # Convert single int to list for consistency

    all_metrics_val = []

    for current_k in k:
        user_recommendations = {}
        for user_id in tqdm(all_users, desc=f'applying knn for {run_name} with k={current_k}'):
            history = user_history[user_id]
            user_vector = user_embs[user_id]
            _, indices = index.search(np.array([user_vector]), current_k + len(history))
            recommendations = [idx for idx in indices[0] if idx not in history][:current_k]
            user_recommendations[user_id] = recommendations
        df = dict_to_pandas(user_recommendations)

        os.makedirs('metrics', exist_ok=True)
        metrics_val = calc_metrics(val, df, current_k)
        metrics_val = metrics_val.apply(mean_confidence_interval)

        # Adjust column names and indices if multiple k values
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

def knn(model_names, suffix, k, mode='val', use_lyrics=False):
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/validation.pqt')
    test = pd.read_parquet('data/test.pqt')
    if isinstance(model_names, str):
        return calc_knn(model_names, train, val, test, mode, suffix, k)
    else:
        return pd.concat([calc_knn(model_name, train, val, test, mode, suffix, k, use_lyrics) for model_name in model_names])
