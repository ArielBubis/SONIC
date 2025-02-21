from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Dict
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch

from SONIC.ROUGE.bert4rec import BERT4Rec
from SONIC.ROUGE.datasets import (
    MaskedLMDataset,
    MaskedLMPredictionDataset,
    PaddingCollateFn,
    load_embeddings,
    prepare_data,
)
from SONIC.CREAM.sonic_utils import (
    calc_metrics,
    dict_to_pandas,
    mean_confidence_interval,
)
@dataclass
class TrainingConfig:
    """
    Basic configuration for training the BERT4Rec model.
    """
    model_name: str = "BERT4Rec"
    max_seq_len: int = 128
    batch_size: int = 128
    num_epochs: int = 200
    lr: float = 0.001
    weight_decay: float = 0.01
    patience_threshold: int = 20
    grad_clip_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tie_weights: bool = True
    init_std: float = 0.02


@dataclass
class DataConfig:
    """
    Basic configuration for data paths and columns.
    """
    train_path: str
    val_path: str
    test_path: str
    embeddings_path: str
    user_col: str = "user_id"
    item_col: str = "track_id"
    time_col: str = "timestamp"
    mode: str = "val"


class BERT4RecTrainer:
    """
    Trainer class for BERT4Rec model.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        run_name: str,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.run_name = run_name

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs, eta_min=1e-6
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100).to(config.device)
        self.scaler = torch.amp.GradScaler("cuda")

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        os.makedirs(f"checkpoints/{config.model_name}", exist_ok=True)
        self.train_losses = []
        self.val_losses = []

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }
        path = f"checkpoints/{self.config.model_name}/{self.run_name}_best.pt"
        torch.save(checkpoint, path)
        # print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> float:
        """
        Load model checkpoint.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            float: Validation loss from the checkpoint.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["val_loss"]

    def _train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                )

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.detach().cpu().item()

        return total_loss / len(self.train_loader)

    def _evaluate(self) -> float:
        """
        Evaluate the model.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                )
                total_loss += loss.item()

        return total_loss / len(self.eval_loader)

    def train(self, save_dir: str, run_name: str, last_epoch: int = 0) -> None:
        """
        Main training loop.

        Args:
            save_dir (str): Directory to save checkpoints.
            run_name (str): Name of the run.
            last_epoch (int, optional): Last epoch number. Defaults to 0.
        """

        log_file = save_dir / f"{run_name}_training_log.txt"
        best_val_loss = float('inf')
        patience = self.config.patience_threshold
        patience_counter = 0
        min_epochs = 10
        # Clear the log file if it exists

        def log_message(message: str):
            with open(log_file, "a") as f:
                f.write(message + "\n")

        # log_message(self.config)
        log_message(str(self.config))

        for epoch in tqdm(range(last_epoch, self.config.num_epochs), desc=f"Training {run_name}"):
            log_message(f"Epoch {epoch}: Starting training")
            # Training phase
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            log_message(f"Train Loss = {train_loss:.4f}")
            # Validation phase
            val_loss = self._evaluate()
            self.val_losses.append(val_loss)
            loss_diff = train_loss - val_loss
            log_message(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Diff = {loss_diff:.4f}")

            self.scheduler.step()
            log_message(f"Loss diff = {loss_diff:.4f}")
            
            # Early warning for overfitting
            if abs(loss_diff) > 0.1:  # Only warn if difference is significant
                if loss_diff > 0.3:
                    log_message(f"Warning: Large train-val gap (possible overfitting): {loss_diff:.4f}")
                elif loss_diff < -0.3:
                    log_message(f"Warning: Validation loss higher than training (possible underfitting): {loss_diff:.4f}")


            # Model checkpoint handling
            if val_loss < self.best_val_loss:
                # tqdm.write(
                #     f"New best model at epoch {epoch} with val loss {val_loss:.4f}, train loss {train_loss:.4f}"
                # )
                improvement = (best_val_loss - val_loss) / best_val_loss
                log_message(
                    f"New best model at epoch {epoch} with val loss {val_loss:.4f}, train loss {train_loss:.4f},Validation improved by {improvement:.2%}"
                )
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping check
            if patience_counter >= patience and epoch >= min_epochs:
                tqdm.write(f"Early stopping triggered at epoch {epoch}")
                log_message(f"Early stopping triggered at epoch {epoch}")
                break

    def generate_recommendations(
        self, pred_loader: DataLoader, user_history: Dict[int, list], k: int = 100
    ) -> Dict[int, list]:
        """
        Generate recommendations for users.

        Args:
            pred_loader (DataLoader): DataLoader for prediction.
            user_history (Dict[int, list]): User history.
            k (int, optional): Number of recommendations to generate. Defaults to 100.

        Returns:
            Dict[int, list]: User recommendations.
        """
        self.model.eval()
        user_recommendations = {}

        with torch.no_grad():
            for batch in tqdm(pred_loader,desc=f"Generating recommendations for k:{k}"):
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                user_ids = batch["user_id"].cpu().numpy()

                outputs = self.model(input_ids, attention_mask)
                seq_lengths = attention_mask.sum(dim=1).long()

                last_item_logits = torch.stack(
                    [outputs[i, seq_lengths[i] - 1, :] for i in range(len(seq_lengths))]
                )
                last_item_logits = last_item_logits[
                    :, :-2
                ]  # Remove mask and padding tokens
                scores, preds = torch.sort(last_item_logits, descending=True)
                preds = preds.cpu().numpy()

                for user_id, item_ids in zip(user_ids, preds):
                    history = user_history.get(user_id, [])
                    recs = [item_id for item_id in item_ids if item_id not in history][
                        :k
                    ]
                    user_recommendations[user_id] = recs

        return user_recommendations

def plot_losses(trainer: BERT4RecTrainer, run_name: str):
    """
    Plot training and validation losses.

    Args:
        trainer (BERT4RecTrainer): Trainer instance.
        run_name (str): Name of the run.
    """
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Save the plot to the plots directory
    plot_path = plots_dir / f'{run_name}_loss_plot.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot to {plot_path}")


def calc_bert(model_name, train, val, test, mode, suffix, k, max_seq_len=128):
    """
    Calculate recommendations using BERT4Rec model.

    Args:
        model_name (str): Name of the model.
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        mode (str): Mode of operation ('val' or 'test').
        suffix (str): Suffix for the run name.
        k (int or list): Number of recommendations to generate.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        pd.DataFrame: Validation metrics.
    """
    run_name = f"bert4rec_{model_name}"
    train, val, user_history, ie = prepare_data(train, val, test)
    embs = load_embeddings(model_name, train, ie)
    special_embs = np.random.normal(0.0, 0.02, size=(2, embs.shape[1]))
    embs = np.concatenate([embs, special_embs], axis=0)
    # log for the model
    print(f"Using {model_name} embeddings")
    item_count = train.item_id.nunique() + 2
    train_dataset = MaskedLMDataset(
        train,
        masking_value=item_count - 2,
        max_length=max_seq_len,
        mlm_probability=0.2,
        force_last_item_masking_prob=0,
    )
    eval_dataset = MaskedLMDataset(
        val,
        masking_value=item_count - 2,
        max_length=max_seq_len,
        mlm_probability=0.2,
        force_last_item_masking_prob=0,
    )
    pred_dataset = MaskedLMPredictionDataset(
        val, 
        masking_value=item_count - 2, 
        max_length=50, 
        validation_mode=True
    )

    # Create directories for saving metrics
    os.makedirs("metrics", exist_ok=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        collate_fn=PaddingCollateFn(padding_value=item_count - 1),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=PaddingCollateFn(padding_value=item_count - 1),
    )
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        collate_fn=PaddingCollateFn(padding_value=item_count - 1),
    )

    model, trainer = train_model(
        train_loader=train_loader,
        eval_loader=eval_loader,
        pred_loader=pred_loader,
        user_history=user_history,
        item_embeddings=embs,
        item_count=item_count,
        embedding_name=model_name,
    )

    # plot_losses(trainer, run_name)

    all_metrics_val = []
    k_values = [k] if isinstance(k, int) else k
    for current_k in k_values:
        # Generate recommendations
        user_recommendations = trainer.generate_recommendations(
            pred_loader=pred_loader,
            user_history=user_history,
            k=current_k
        )
        # Calculate metrics
        df = dict_to_pandas(user_recommendations)
        metrics_val = calc_metrics(val, df, current_k)
        metrics_val = metrics_val.apply(mean_confidence_interval)

        # Format metric names
        if len(k_values) > 1:
            metrics_val.columns = [
                f'{col.split("@")[0]}@k' for col in metrics_val.columns
            ]
            metrics_val.index = [f"mean at k={current_k}", f"CI at k={current_k}"]

        all_metrics_val.append(metrics_val)

    # Combine metrics for different k values
    metrics_val_concat = (
        pd.concat(all_metrics_val, axis=0) if len(k_values) > 1 else all_metrics_val[0]
    )

    # Save results
    metrics_val_concat.to_csv(f"metrics/{run_name}_val.csv")
    print(f"Saved metrics to metrics/{run_name}_val.csv")

    return metrics_val_concat


def train_model(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    pred_loader: DataLoader,
    user_history: Dict[int, list],
    item_embeddings: np.ndarray,
    item_count: int,
    embedding_name: str,
    save_dir: str = "bert_model",
) -> BERT4Rec:
    """
    Train the final model with the best parameters for 200 epochs.

    Args:
        train_loader: Training data loader
        eval_loader: Validation data loader
        pred_loader: Prediction data loader
        user_history: Dictionary of user histories
        item_embeddings: Pre-computed item embeddings
        item_count: Total number of items
        embedding_name: Name of the embedding being used
        save_dir: Directory to save model checkpoints and logs
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load best parameters, these are taken from the original article, with slight tweaks

    best_params = {
        "model_params": {
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 256,
        },
        "train_params": {"lr": 0.001, "weight_decay": 0.005, "batch_size": 128},
    }
    # Create model config
    model_config = {
        "vocab_size": item_count,
        "max_position_embeddings": 200,
        **best_params["model_params"],
    }

    # Create training config
    train_config = TrainingConfig(
        model_name="BERT4Rec",
        max_seq_len=128,
        num_epochs=200,
        **best_params["train_params"],
    )
    print(train_config)

    # Initialize model
    model = BERT4Rec(
        vocab_size=item_count,
        bert_config=model_config,
        precomputed_item_embeddings=item_embeddings,
        # original_embedding_dim=original_embedding_dim,
        padding_idx=item_count - 1,
        add_head=True,
    )

    # Create unique run name
    run_name = f"{embedding_name}_{save_dir}"

    # Initialize trainer
    trainer = BERT4RecTrainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_name=run_name,
    )

    # Create log file
    log_file = save_dir / f"{run_name}_training_log.txt"
    with open(log_file, "w") as f:
        pass

    def log_message(message: str):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    final_model_path = save_dir / f"{run_name}_final.pt"

    # Check if final model exists in the save directory, if so, load it and skip training
    if final_model_path.exists():
        print(f"Final model found at {final_model_path}. Loading model...")
        checkpoint = torch.load(final_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.best_val_loss = checkpoint["best_val_loss"]
        log_message(f"Final model loaded from {final_model_path}")

    else:
        # Check if checkpoint exists
        checkpoint_path = f"checkpoints/BERT4Rec/{run_name}_best.pt"
        if checkpoint_path.exists():
            print(f"Checkpoint found at {checkpoint_path}. Loading checkpoint...")
            trainer.load_checkpoint(checkpoint_path)
            log_message(f"Checkpoint loaded from {checkpoint_path}")

        # Train model from scratch
        else:
            log_message(f"Starting training with best parameters:")
            log_message(
                f"Model parameters: {json.dumps(best_params['model_params'], indent=2)}"
            )
            log_message(
                f"Training parameters: {json.dumps(best_params['train_params'], indent=2)}"
            )

            # Train model
            trainer.train(save_dir,run_name)

            try:
                plot_losses(trainer, run_name)
            except Exception as e:
                pass
            # Save final model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "train_config": asdict(train_config),
                    "best_val_loss": trainer.best_val_loss,
                },
                final_model_path,
            )
            
            log_message(f"\nTraining completed!")
            log_message(f"Best validation loss: {trainer.best_val_loss:.4f}")

            log_message(f"\nFinal model saved to {final_model_path}")

    return model, trainer


def bert_train(model_names, suffix, k, mode="val", max_seq_len=128):
    """
    Train and evaluate the BERT4Rec model.

    Args:
        model_names (str or list): Name(s) of the model(s) to train.
        suffix (str): Suffix for the run name.
        k (int or list): Number of recommendations to generate.
        mode (str, optional): Mode of operation ('val' or 'test'). Defaults to 'val'.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 128.

    Returns:
        pd.DataFrame: Validation metrics.
    """
    
    train = pd.read_parquet("data/train.pqt")
    val = pd.read_parquet("data/validation.pqt")
    test = pd.read_parquet("data/test.pqt")
    
    if isinstance(model_names, str):
        return calc_bert(model_names, train, val, test, mode, suffix, k, max_seq_len)
    else:
        return pd.concat(
            [
                calc_bert(model_name, train, val, test, mode, suffix, k, max_seq_len)
                for model_name in model_names
            ]
        )

