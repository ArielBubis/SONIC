class BERT4RecEvaluator:
    """Evaluator class for BERT4Rec model."""
    def __init__(
        self,
        model: BERT4Rec,
        config: TrainingConfig,
        pred_loader: DataLoader
    ):
        self.model = model.to(config.device)
        self.config = config
        self.pred_loader = pred_loader

    def generate_recommendations(self, k: int = 100) -> Dict[int, list]:
        """Generate recommendations for all users."""
        self.model.eval()
        user_recommendations = {}
        
        with torch.no_grad():
            for batch in tqdm(self.pred_loader, desc="Generating recommendations"):
                batch_recommendations = self._generate_batch_recommendations(batch, k)
                user_recommendations.update(batch_recommendations)
                
        return user_recommendations

    def _generate_batch_recommendations(self, batch, k: int) -> Dict[int, list]:
        """Generate recommendations for a batch of users."""
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)
        user_ids = batch['user_id'].numpy()
        histories = batch['full_history']
        
        outputs = self.model(input_ids, attention_mask)
        seq_lengths = attention_mask.sum(dim=1).long()
        
        last_item_logits = torch.stack([
            outputs[i, seq_lengths[i] - 1, :]
            for i in range(len(seq_lengths))
        ])
        last_item_logits = last_item_logits[:, :-2]
        scores, preds = torch.sort(last_item_logits, descending=True)
        preds = preds.cpu().numpy()
        
        batch_recommendations = {}
        for user_id, history, recommendations in zip(user_ids, histories, preds):
            filtered_recs = [
                item_id for item_id in recommendations
                if item_id not in history
            ][:k]
            batch_recommendations[user_id] = filtered_recs
            
        return batch_recommendations

    def evaluate_recommendations(
        self,
        recommendations: Dict[int, list],
        ground_truth: pd.DataFrame,
        k: int = 100
    ) -> pd.DataFrame:
        """Evaluate recommendations using various metrics."""
        from rs_metrics import hitrate, mrr, recall, ndcg
        
        metrics = pd.DataFrame()
        df_recommendations = dict_to_pandas(recommendations)
        
        metrics[f'HitRate@{k}'] = hitrate(ground_truth, df_recommendations, k=k, apply_mean=False)
        metrics[f'MRR@{k}'] = mrr(ground_truth, df_recommendations, k=k, apply_mean=False)
        metrics[f'Recall@{k}'] = recall(ground_truth, df_recommendations, k=k, apply_mean=False)
        metrics[f'NDCG@{k}'] = ndcg(ground_truth, df_recommendations, k=k, apply_mean=False)
        
        return metrics