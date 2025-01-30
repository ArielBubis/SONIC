from poplib import CR
from click import Option
import torch
from . import CREAM
from . import TAILS
from . import ROUGE

import os
import typer
from rich.console import Console
from rich.prompt import Prompt
from typing import Optional, List

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
    CREAM.split.split_data(interactions_file, sep, start_date, test_date, end_date)
    
    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')

@app.command()
def run_model(
    model_name: str = typer.Option(..., "--model-name", help="Name of the model to run"),
    embedding: Optional[List[str]] = typer.Option(['mfcc_104'], "--embedding", help="Name of the embedding model"),
    mode: str = typer.Option('val', "--mode", help="Mode for running the model (val or test)"),
    suffix: str = typer.Option('cosine', "--suffix", help="Suffix for the model name"),
    k: Optional[List[int]] = typer.Option([50], "--k", help="List of k nearest neighbors to retrieve"),
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
        ROUGE.knn.knn(embedding, suffix, k, mode)
    
    
    if profile:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')


if __name__ == "__main__":
    app()
