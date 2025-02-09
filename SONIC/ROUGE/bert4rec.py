import os
import torch
import pandas as pd
import logging  # Add logging import
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed

# Configure logging to suppress specific warnings and errors
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class BERT4RecRecommender:
    def __init__(self, config_file='bert4rec.yaml'):
        """Initialize BERT4Rec Recommender with given config file."""
        self.config = Config(model=BERT4Rec, config_file_list=[config_file])
        init_seed(self.config['seed'], self.config['reproducibility'])

        # Load dataset
        self.dataset = create_dataset(self.config)
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)

        # Initialize model
        self.model = BERT4Rec(self.config, self.train_data.dataset).to(self.config['device'])
        self.trainer = Trainer(self.config, self.model)

    def train(self):
        """Train the BERT4Rec model."""
        best_valid_score, best_valid_result = self.trainer.fit(self.train_data, self.valid_data)
        print("Best Validation Result:", best_valid_result)
        self.save_model()
        return best_valid_result

    def save_model(self, save_path='models/bert4rec.pth'):
        """Save the trained model."""
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    def load_model(self, load_path='models/bert4rec.pth'):
        """Load a pre-trained BERT4Rec model."""
        self.model.load_state_dict(torch.load(load_path, map_location=self.config['device']))
        self.model.eval()
        print(f"Model loaded from {load_path}")

    def generate_recommendations(self, user_interactions, top_k=10):
        """
        Generate recommendations for users.

        Args:
            user_interactions (pd.DataFrame): User interaction history.
            top_k (int): Number of recommendations per user.

        Returns:
            dict: User ID -> List of recommended item IDs.
        """
        user_history = user_interactions.groupby('user_id')['track_id'].apply(list).to_dict()
        recommendations = {}

        with torch.no_grad():
            for user_id, history in user_history.items():
                seq = torch.tensor(history[-self.config['max_seq_length']:], dtype=torch.long).to(self.config['device'])
                seq = seq.unsqueeze(0)  # Add batch dimension
                scores = self.model.full_sort_predict(seq)
                top_items = torch.argsort(scores, descending=True)[0, :top_k].cpu().numpy()
                recommendations[user_id] = top_items.tolist()

        return recommendations
