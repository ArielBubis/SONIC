import os
import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertModel
from tqdm.auto import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split


class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, bert_config, precomputed_item_embeddings=None, add_head=True,
                 tie_weights=True, padding_idx=-1, init_std=0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        if precomputed_item_embeddings is not None:
            precomputed_item_embeddings = torch.from_numpy(precomputed_item_embeddings.astype(np.float32))
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
            self.head = nn.Linear(bert_config['hidden_size'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.item_embeddings.weight

        if precomputed_item_embeddings is None:
            self.item_embeddings.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.item_embeddings.weight.data[self.padding_idx].zero_()

    def freeze_item_embs(self, flag):
        self.item_embeddings.weight.requires_grad = flag

    def forward(self, input_ids, attention_mask):
        embeds = self.item_embeddings(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask
        )
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs


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


def run_bert4rec(train, val, test, mode='val', k=50):
    train, val, user_history, ie = prepare_data(train, val, test, mode)

    # Initialize BERT4Rec model
    bert_config = {
        'hidden_size': 768,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 3072,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'initializer_range': 0.02
    }
    model = BERT4Rec(vocab_size=len(ie.classes_), bert_config=bert_config)

    # Train the model (placeholder for training logic)
    # TODO: Implement training loop

    # Generate recommendations
    user_recommendations = {}
    for user_id in tqdm(val.user_id.unique(), desc="Generating recommendations"):
        history = user_history[user_id]
        input_ids = torch.tensor(list(history), dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids, attention_mask)
        scores = outputs.squeeze().detach().numpy()
        recommendations = np.argsort(-scores)[:k]
        user_recommendations[user_id] = recommendations

    # Evaluate recommendations
    df = dict_to_pandas(user_recommendations)
    metrics = calc_metrics(val, df, k)
    metrics = metrics.apply(mean_confidence_interval)

    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    metrics.to_csv(f'metrics/bert4rec_{mode}.csv')

    return metrics


def bert4rec(model_names, suffix, k, mode='val'):
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/validation.pqt')
    test = pd.read_parquet('data/test.pqt')
    return run_bert4rec(train, val, test, mode, k)