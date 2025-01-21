import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split

class ShallowEmbeddingModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, precomputed_user_embeddings=None, emb_dim_out=300):
        super(ShallowEmbeddingModel, self).__init__()
        self.emb_dim_in = emb_dim_in

        if precomputed_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in)
        else:
            precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings)
            assert precomputed_user_embeddings.size(1) == emb_dim_in
            self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings)

        if precomputed_item_embeddings is None:
            self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in)
        else:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings)
            assert precomputed_item_embeddings.size(1) == emb_dim_in
            self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings)

        self.model = nn.Sequential(
            nn.Linear(self.emb_dim_in, emb_dim_out),
            nn.ReLU()
        )

        self.cossim = torch.nn.CosineSimilarity()

    def freeze_item_embs(self, flag):
        self.item_embeddings.weight.requires_grad = flag

    def freeze_user_embs(self, flag):
        self.user_embeddings.weight.requires_grad = flag

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)

        user_embeds = self.model(user_embeds)
        item_embeds = self.model(item_embeds)

        scores = self.cossim(user_embeds, item_embeds)
        return scores

    def extract_embeddings(self):
        user_embeddings = self.user_embeddings.weight.data
        item_embeddings = self.item_embeddings.weight.data

        with torch.no_grad():
            user_embeddings = self.model(user_embeddings).cpu().numpy()
            item_embeddings = self.model(item_embeddings).cpu().numpy()

        user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        return user_embeddings, item_embeddings

class ShallowInteractionModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim_in, precomputed_item_embeddings=None, precomputed_user_embeddings=None, emb_dim_out=300):
        super(ShallowInteractionModel, self).__init__()
        self.emb_dim_in = emb_dim_in

        if precomputed_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, self.emb_dim_in)
        else:
            precomputed_user_embeddings = torch.from_numpy(precomputed_user_embeddings)
            assert precomputed_user_embeddings.size(1) == emb_dim_in
            self.user_embeddings = nn.Embedding.from_pretrained(precomputed_user_embeddings)

        if precomputed_item_embeddings is None:
            self.item_embeddings = nn.Embedding(num_items, self.emb_dim_in)
        else:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings)
            assert precomputed_item_embeddings.size(1) == emb_dim_in
            self.item_embeddings = nn.Embedding.from_pretrained(precomputed_item_embeddings)

        self.model = nn.Sequential(
            nn.Linear(self.emb_dim_in, emb_dim_out),
            nn.ReLU()
        )

        self.fc = nn.Linear(emb_dim_out, 1)

    def freeze_item_embs(self, flag):
        self.item_embeddings.weight.requires_grad = flag

    def freeze_user_embs(self, flag):
        self.user_embeddings.weight.requires_grad = flag

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)

        user_embeds = self.model(user_embeds)
        item_embeds = self.model(item_embeds)

        hadamard_product = user_embeds * item_embeds
        scores = self.fc(hadamard_product)

        return torch.sigmoid(scores).squeeze()

    def extract_embeddings(self):
        user_embeddings = self.user_embeddings.weight.data
        item_embeddings = self.item_embeddings.weight.data

        with torch.no_grad():
            user_embeddings = self.model(user_embeddings).cpu().numpy()
            item_embeddings = self.model(item_embeddings).cpu().numpy()

        user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        return user_embeddings, item_embeddings

def load_embeddings(model_name, train, ie):
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

def snn(model_name, train, val, test, mode='val', emb_dim_out=300):
    train, val, user_history, ie = prepare_data(train, val, test, mode)
    item_embs = load_embeddings(model_name, train, ie)

    num_users = train['user_id'].nunique()
    num_items = train['item_id'].nunique()
    emb_dim_in = item_embs.shape[1]

    model = ShallowEmbeddingModel(num_users, num_items, emb_dim_in, precomputed_item_embeddings=item_embs, emb_dim_out=emb_dim_out)
    user_embs, item_embs = model.extract_embeddings()

    # Further processing and evaluation can be done here

    return user_embs, item_embs