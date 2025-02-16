from poplib import CR
from click import Option
from sympy import use
import torch
from . import CREAM
from . import TAILS
from . import ROUGE

import os
import typer
from rich.console import Console
from rich.prompt import Prompt
from typing import Optional, List

import os
from ROUGE.bert4rec import (
    BERT4Rec, 
    BERT4RecTrainer, 
    BERT4RecEvaluator,
    TrainingConfig,
    MaskedLMDataset,
    PredictionDataset,
    PaddingCollateFn
)
app = typer.Typer()
console = Console()

def run_embed_task(audio_path: str, model_name: str, batch_size: int, np_precision: str):
    """
    Execute the embedding task with validated arguments.
    """
    CREAM.utils.setup_logging('audio_embedding.log')
    if not os.path.exists(audio_path):
        console.print(f"[bold red]Audio directory {audio_path} does not exist.[/bold red]")
        raise typer.Exit()

    CREAM.utils.setup_logging('audio_embedding.log')
    if "mfcc" in model_name.lower():
        emb = TAILS.MFCC.MFCCEmbedder(batch_size=batch_size).get_embeddings(audio_path)
    
    elif "m2v" in model_name.lower():
        emb = TAILS.M2V.M2VEmbedder(batch_size=batch_size).get_embeddings(audio_path)
    
    elif "mert" in model_name.lower():
        emb = TAILS.MERT.MERTEmbedder(batch_size=batch_size).get_embeddings(audio_path)

    elif "vit" in model_name.lower():
        emb = TAILS.ViT.ViTEmbedder(batch_size=batch_size, model_name=model_name).get_embeddings(audio_path)

    elif "lyrical" in model_name.lower():
        emb = TAILS.Lyrical.LyricalEmbedder().get_embeddings(audio_path)

    # elif "musicfm" in model_name.lower():
    #     emb = TAILS.MusicFM.MusicFMEmbedder(batch_size=batch_size).get_embeddings(audio_path)

    # elif "musicnn" in model_name.lower():
    #     emb = TAILS.MusiCNN.MusicNNEmbedder(batch_size=batch_size).get_embeddings(audio_path)
    #     raise typer.Exit()

    elif "encodecmae" in model_name.lower():
        num_gpus = torch.cuda.device_count()
        if num_gpus>0:
            emb = TAILS.EncodecMAE.EncodecMAEEmbedder(batch_size=batch_size).get_embeddings(audio_path)
        else:
            console.print("[bold red] CUDA is required to run this model [/bold red]")
            raise typer.Exit()

    else:
        console.print(f"[bold red]Unsupported model type: {model_name}[/bold red]")
        raise typer.Exit()

    CREAM.io.save_embeddings(emb, f"embeddings/{model_name}.pqt", kwargs={'np_type': np_precision})

    console.print(f"[green]Embeddings saved to `{os.getcwd()}/embeddings/{model_name}.pqt`[/green]")

@app.command()
def embed(
    audio_dir: str = typer.Option(..., "--audio-dir", help="Directory containing audio files"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for processing audio files"),
    model_type: str = typer.Option(..., "--model-type", help="Model type (e.g., MFCC, ViT, MERT)"),
    np_precision: str = typer.Option('float16', "--np-precision", help="Numpy precision for saving embeddings"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """
    Embed audio files using the specified model type.
    """
    if profile:
        profiler = CREAM.utils.start_profiler()

    run_embed_task(audio_dir, model_type, batch_size, np_precision)

    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

@app.command()
def tui():
    """
    Launch the interactive TUI for embedding tasks.
    """
    console.print("[bold blue]Welcome to the Audio Processing CLI![/bold blue]")
    task = Prompt.ask("Choose a task", choices=["embed"], default="embed")

    if task == "embed":
        audio_dir = Prompt.ask("Enter the path to the audio directory")
        model_type = Prompt.ask("Enter the model type (e.g., MFCC, ViT, MusiCNN)")
        
        run_embed_task(audio_dir, model_type)

@app.command()
def data_split(
    interactions_file: str = typer.Option(..., "--interactions-file", help="Path to the interactions file"),
    sep: str = typer.Option(',', "--sep", help="Delimiter used in the interactions file (csv only)"),
    exclude: Optional[str] = typer.Option(None, "--exclude-file", help="Path to the exclude file"),
    start_date: str = typer.Option(CREAM.split.START_DATE, "--start-date", help="Start date for splitting data"),
    end_date: str = typer.Option(CREAM.split.TEST_DATE, "--end-date", help="End date for splitting data"),
    test_date: str = typer.Option(CREAM.split.END_DATE, "--test-date", help="Test date for splitting data"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """
    Split the interactions data into training and validation sets based on the specified date range.
    """
    if profile:
        profiler = CREAM.utils.start_profiler()
    if not os.path.exists(interactions_file):
        console.print(f"[bold red]Interactions file {interactions_file} does not exist.[/bold red]")
        raise typer.Exit()
    CREAM.utils.setup_logging('data_split.log')
    CREAM.split.split_data(interactions_file, sep, start_date, test_date, end_date, exclude_file=exclude)
    
    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

@app.command()
def run_model(
    model_name: str = typer.Option(..., "--model-name", help="Name of the model to run"),
    embedding: Optional[List[str]] = typer.Option(['mfcc_104'], "--embedding", help="Name of the embedding model"),
    mode: str = typer.Option('val', "--mode", help="Mode for running the model (val or test)"),
    suffix: str = typer.Option('cosine', "--suffix", help="Suffix for the model name"),
    k: Optional[List[int]] = typer.Option([50], "--k", help="List of k nearest neighbors to retrieve"),
    use_lyrics: bool = typer.Option(False, "--use-lyrics", help="Use lyrics embeddings"),
    use_metadata: bool = typer.Option(False, "--use-metadata", help="Use track metadata"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """
    Run the specified model on the provided training, validation, and test data.
    """
    if profile:
        profiler = CREAM.utils.start_profiler()
    if not os.path.exists('data/train.pqt') or not os.path.exists('data/validation.pqt') or not os.path.exists('data/test.pqt'):
        console.print(f"[bold red]One or more data files do not exist.[/bold red]")
        raise typer.Exit()
    CREAM.utils.setup_logging('model_run.log')
    if model_name == 'knn':
        ROUGE.knn.knn(embedding, suffix, k, mode, use_lyrics=use_lyrics, use_metadata=use_metadata)
    elif model_name == 'snn':
        ROUGE.snn.snn(embedding, suffix, k, mode)
    
    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

@app.command()
def find_corrupt(
    audio_dir: str = typer.Option(..., "--audio-dir", help="Directory containing audio files"),
    loudness_threshold: float = typer.Option(..., "--loudness-threshold", help="Loudness threshold for corrupt audio files"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """
    Find and remove corrupt audio files in the specified directory.
    """
    if profile:
        profiler = CREAM.utils.start_profiler()
    CREAM.utils.setup_logging('find_corrupt.log')
    corrupt = CREAM.utils.scan_corrupt_audio(audio_dir, loudness_threshold)

    if corrupt:
        CREAM.io.save_exclude_list(corrupt)
        console.print(f"[bold red]Corrupt audio files saved to `exclude.pqt`[/bold red]")
    
    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

@app.command()
def train_bert(
    train_file: str = typer.Option(..., "--train-file", help="Path to training data parquet file"),
    val_file: str = typer.Option(..., "--val-file", help="Path to validation data parquet file"),
    embedding_model: str = typer.Option(..., "--embedding-model", help="Name of embedding model to use"),
    max_seq_len: int = typer.Option(128, "--max-seq-len", help="Maximum sequence length"),
    batch_size: int = typer.Option(128, "--batch-size", help="Training batch size"),
    num_epochs: int = typer.Option(200, "--num-epochs", help="Number of training epochs"),
    lr: float = typer.Option(0.001, "--lr", help="Learning rate"),
    hidden_dim: int = typer.Option(256, "--hidden-dim", help="Hidden dimension size"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """Train BERT4Rec model with specified parameters."""
    if profile:
        profiler = CREAM.utils.start_profiler()

    try:
        # Load data
        train_data = pd.read_parquet(train_file)
        val_data = pd.read_parquet(val_file)
        
        # Load embeddings
        item_embeddings = CREAM.utils.load_embeddings(embedding_model, train_data)
        
        # Create config
        config = TrainingConfig(
            model_name="BERT4Rec",
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr
        )
        
        # Initialize datasets
        item_count = train_data.item_id.nunique() + 2
        train_dataset = MaskedLMDataset(
            train_data,
            max_length=max_seq_len,
            masking_value=item_count-2
        )
        val_dataset = MaskedLMDataset(
            val_data,
            max_length=max_seq_len,
            masking_value=item_count-2
        )
        
        # Create dataloaders
        collate_fn = PaddingCollateFn(padding_value=item_count-1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize model
        model_config = {
            'vocab_size': item_count,
            'max_position_embeddings': max(200, max_seq_len),
            'hidden_size': hidden_dim,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'intermediate_size': hidden_dim * 4
        }
        
        model = BERT4Rec(
            vocab_size=item_count,
            bert_config=model_config,
            precomputed_item_embeddings=item_embeddings,
            padding_idx=item_count-1
        )
        
        # Initialize trainer
        trainer = BERT4RecTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Train model
        best_val_loss = trainer.train()
        console.print(f"[green]Training completed with best validation loss: {best_val_loss:.4f}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error during training: {str(e)}[/bold red]")
        raise typer.Exit(1)

    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

@app.command()
def evaluate_bert(
    model_path: str = typer.Option(..., "--model-path", help="Path to trained model checkpoint"),
    test_file: str = typer.Option(..., "--test-file", help="Path to test data parquet file"),
    k: List[int] = typer.Option([10, 50, 100], "--k", help="List of k values for evaluation"),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size for evaluation"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """Evaluate trained BERT4Rec model and generate recommendations."""
    if profile:
        profiler = CREAM.utils.start_profiler()

    try:
        # Load test data
        test_data = pd.read_parquet(test_file)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path)
        config = TrainingConfig(batch_size=batch_size)
        
        model = BERT4Rec(
            vocab_size=checkpoint['model_config']['vocab_size'],
            bert_config=checkpoint['model_config'],
            padding_idx=checkpoint['model_config']['vocab_size'] - 1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create prediction dataset and loader
        pred_dataset = PredictionDataset(
            test_data,
            max_length=config.max_seq_len,
            masking_value=test_data.item_id.nunique()
        )
        pred_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=PaddingCollateFn()
        )
        
        # Initialize evaluator
        evaluator = BERT4RecEvaluator(
            model=model,
            config=config,
            pred_loader=pred_loader
        )
        
        # Generate and evaluate recommendations
        for current_k in k:
            recommendations = evaluator.generate_recommendations(k=current_k)
            metrics = evaluator.evaluate_recommendations(
                recommendations,
                test_data,
                k=current_k
            )
            
            # Save results
            os.makedirs('metrics', exist_ok=True)
            metrics.to_csv(f'metrics/bert4rec_k{current_k}.csv')
            
            # Save recommendations
            os.makedirs('preds', exist_ok=True)
            recommendations_df = CREAM.utils.dict_to_pandas(recommendations)
            recommendations_df.to_parquet(f'preds/bert4rec_k{current_k}.pqt')
            
            console.print(f"[green]Evaluation completed for k={current_k}[/green]")
            console.print(metrics.mean().to_dict())

    except Exception as e:
        console.print(f"[bold red]Error during evaluation: {str(e)}[/bold red]")
        raise typer.Exit(1)

    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

if __name__ == "__main__":
    app()