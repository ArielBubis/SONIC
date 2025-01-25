import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from rs_metrics import hitrate, mrr, precision, recall, ndcg
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval
from SONIC.CREAM.dataset import InteractionDataset

def train_bert(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, checkpoint_dir):
    best_val_loss = float('inf')
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, attention_mask = batch['input_ids'].to(device), batch['labels'].to(device), batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))

def evaluate_bert(model, test_loader, device, k=100):
    model.eval()
    user_recommendations = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_ids = batch['user_id']

            outputs = model(input_ids, attention_mask)
            seq_lengths = attention_mask.sum(dim=1).long()
            last_item_logits = torch.stack([outputs[i, seq_lengths[i] - 1, :] for i in range(len(seq_lengths))])
            last_item_logits = last_item_logits[:, :-2]  # Remove mask and padding tokens
            scores, preds = torch.sort(last_item_logits, descending=True)
            preds = preds.cpu().numpy()
            user_history = train.groupby('user_id', observed=False)['item_id'].agg(set).to_dict()
            for user_id, item_ids in zip(user_ids, preds):
                user_id = user_id.item()
                history = user_history[user_id]
                recs = [item_id for item_id in item_ids if item_id not in history][:k]
                user_recommendations[user_id] = recs

    return user_recommendations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train = pd.read_parquet('data/train.pqt')
    val = pd.read_parquet('data/val.pqt')
    test = pd.read_parquet('data/test.pqt')

    # Initialize model
    model_params = {
        'vocab_size': train['item_id'].nunique() + 2,
        'max_position_embeddings': 200,
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'intermediate_size': 256
    }
    model = BERT4Rec(**model_params).to(device)

    # Train
    train_loader = DataLoader(InteractionDataset(train), batch_size=128, shuffle=True)
    val_loader = DataLoader(InteractionDataset(val), batch_size=128, shuffle=False)
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()
    train_bert(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, checkpoint_dir='checkpoints/bert')

    # Evaluate
    test_loader = DataLoader(InteractionDataset(test), batch_size=128, shuffle=False)
    user_recommendations = evaluate_bert(model, test_loader, device)
    df = dict_to_pandas(user_recommendations)
    df.to_parquet('preds/bert_recommendations.pqt')

    # Calculate metrics
    metrics = calc_metrics(test, df)
    print(metrics)

if __name__ == '__main__':
    main()


import torch
from torch import nn
from transformers import BertConfig, BertModel



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