from torch import device, le
from . import CREAM
from . import TAILS

import os
import typer
from rich.console import Console
from rich.prompt import Prompt

app = typer.Typer()
console = Console()

def validate_args(model_name: str, stride: int | None, level: int | None) -> int | None:
    """
    Validate arguments based on the model type.
    """
    if "vit" not in model_name.lower() and stride is not None:
        console.print("[bold yellow]Warning:[/bold yellow] Stride is only applicable to ViT models. Ignoring stride.")
        stride = None
    elif "vit" not in model_name.lower() and level is None:
        console.print("[bold yellow]Warning:[/bold yellow] Level is only applicable to ViT models. Ignoring level.")
        level = 0
    return stride, level

def run_embed_task(audio_path: str, model_name: str, batch_size: int, stride: int | None = None, level: int = 0):
    """
    Execute the embedding task with validated arguments.
    """
    if not os.path.exists(audio_path):
        console.print(f"[bold red]Audio directory {audio_path} does not exist.[/bold red]")
        raise typer.Exit()

    stride, level = validate_args(model_name, stride, level)

    CREAM.utils.setup_logging('audio_embedding.log')
    if "mfcc" in model_name.lower():
        emb = TAILS.MFCC.MFCCEmbedder(batch_size=batch_size).get_embeddings(audio_path)

    elif "vit" in model_name.lower():
        if stride is None:
            console.print("[bold red]Stride is required for ViT models.[/bold red]")
            raise typer.Exit()
        emb = TAILS.ViT.ViTEmbedder(batch_size=batch_size, stride=stride, level=level).get_embeddings(audio_path)

    elif "musicnn" in model_name.lower():
        console.print("[bold red]MusicNN model is not yet supported.[/bold red]")
        raise typer.Exit()

    else:
        console.print(f"[bold red]Unsupported model type: {model_name}[/bold red]")
        raise typer.Exit()

    CREAM.io.save_embeddings(emb, f"{model_name}_embeddings.csv")
    console.print(f"[green]Embeddings saved to {model_name}_embeddings.csv[/green]")

@app.command()
def embed(
    audio_dir: str = typer.Option(..., "--audio-dir", help="Directory containing audio files"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for processing audio files"),
    model_type: str = typer.Option(..., "--model-type", help="Model type (e.g., MFCC, ViT, MusicNN)"),
    stride: int = typer.Option(None, "--stride", help="Stride length for extracting windows (only for ViT)"),
    level: int = typer.Option(0, "--level", help="Layer level for ViT models"),
    profile: bool = typer.Option(False, "--profile", help="Enable profiling"),
):
    """
    Embed audio files using the specified model type.
    """
    if profile:
        profiler = CREAM.utils.start_profiler()

    run_embed_task(audio_dir, model_type, batch_size, stride, level)

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
        stride = None
        if "vit" in model_type.lower():
            stride = Prompt.ask("Enter the stride length for extracting windows", default="1")
        
        run_embed_task(audio_dir, model_type, int(stride) if stride else None)

if __name__ == "__main__":
    app()
