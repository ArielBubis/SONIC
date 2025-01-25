import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, AdamW
from tqdm.auto import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval, safe_split


class BERT4rec(nn.Module):
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


class SequenceDataset(Dataset):
    def __init__(self, data, max_seq_length=50):
        self.data = data
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['sequence']
        sequence = sequence[-self.max_seq_length:]  # Truncate to max_seq_length
        input_ids = torch.tensor(sequence, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask


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

    # Group interactions into sequences
    train['sequence'] = train.groupby('user_id')['item_id'].apply(lambda x: list(x))
    val['sequence'] = val.groupby('user_id')['item_id'].apply(lambda x: list(x))

    user_history = train.groupby('user_id', observed=False)['item_id'].agg(set).to_dict()
    return train, val, user_history, ie


def train_bert4rec(model, train_loader, val_loader, optimizer, criterion, device, epochs=5):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0

        # Training loop
        for batch in tqdm(train_loader, desc="Training"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Training Loss: {total_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")
        model.train()


def run_bert4rec(train, val, test, mode='val', k=50, epochs=5):
    train, val, user_history, ie = prepare_data(train, val, test, mode)

    # Initialize BERT4Rec model
    bert_config = {
        'hidden_size': 64,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 200,
        'type_vocab_size': 2,
        'initializer_range': 0.02
    }
    model = BERT4Rec(vocab_size=len(ie.classes_), bert_config=bert_config)

    # Prepare data loaders
    train_dataset = SequenceDataset(train)
    val_dataset = SequenceDataset(val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bert4rec(model, train_loader, val_loader, optimizer, criterion, device, epochs=epochs)

    # Generate recommendations
    model.eval()
    user_recommendations = {}
    for user_id in tqdm(val.user_id.unique(), desc="Generating recommendations"):
        history = user_history[user_id]
        input_ids = torch.tensor(list(history), dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        outputs = model(input_ids, attention_mask)
        scores = outputs.squeeze().detach().cpu().numpy()
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


def bert4rec(model_names, suffix, k, mode='val', epochs=5):
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/validation.pqt')
    test = pd.read_parquet('data/test.pqt')
    return run_bert4rec(train, val, test, mode, k, epochs)