import os
import logging
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from SONIC.CREAM.sonic_utils import dict_to_pandas, calc_metrics, mean_confidence_interval

def prepare_data(config_file="bert4rec.yaml"):
    config = Config(model=BERT4Rec, config_file_list=[config_file])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return config, train_data, valid_data, test_data

def train_bert4rec(config_file="bert4rec.yaml"):
    config, train_data, valid_data, _ = prepare_data(config_file)
    model = BERT4Rec(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    logging.info(f"Best validation result: {best_valid_result}")
    return model

def predict_bert4rec(model, test_data, k=50):
    model.eval()
    with torch.no_grad():
        scores = model.full_sort_predict(test_data)
    recommendations = {}
    for user_id in tqdm(test_data.dataset.inter_feat['user_id'].unique(), desc='Generating recommendations'):
        user_scores = scores[user_id].cpu().numpy()
        top_items = np.argsort(user_scores)[::-1][:k]
        recommendations[user_id] = top_items
    return recommendations

def evaluate_bert4rec(model, test_data, k=[10, 20, 50]):
    all_metrics = []
    recommendations = predict_bert4rec(model, test_data, max(k))
    for current_k in k:
        filtered_recommendations = {user: items[:current_k] for user, items in recommendations.items()}
        df = dict_to_pandas(filtered_recommendations)
        os.makedirs('metrics', exist_ok=True)
        metrics = calc_metrics(test_data.dataset.inter_feat, df, current_k)
        metrics = metrics.apply(mean_confidence_interval)
        all_metrics.append(metrics)
    metrics_concat = pd.concat(all_metrics, axis=0) if len(k) > 1 else all_metrics[0]
    metrics_concat.to_csv('metrics/bert4rec_metrics.csv')
    return metrics_concat
