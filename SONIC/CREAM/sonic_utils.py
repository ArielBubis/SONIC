import matplotlib.pyplot as plt
from PIL import Image
import cProfile
import logging
import torch
from torchvision import transforms
import pandas as pd
import os

class SpectrogramDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading spectrograms and ensuring preprocessing.
    """
    def __init__(self, spectograms_dir):
        self.spectograms_dir = spectograms_dir
        self.spectograms = os.listdir(spectograms_dir)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 448)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.spectograms)

    def __getitem__(self, idx):
        spectrogram_path = os.path.join(self.spectograms_dir, self.spectograms[idx])
        spectrogram = Image.open(spectrogram_path).convert("RGB")
        spectrogram = self.transforms(spectrogram)
        return spectrogram_path, spectrogram

def init_dataset(spectograms_dir, batch_size=32):
    """
    Initialize the dataset for the given spectrogram directory.
    """
    dataset = SpectrogramDataset(spectograms_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def save_image(fig, output_filepath):
    """
    Save the figure as an image file.
    """
    fig.canvas.draw()
    
    # Get the RGBA buffer
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    
    # Convert to PIL Image directly from buffer
    image = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)
    
    # Convert RGBA to RGB and save
    image.convert('RGB').save(output_filepath, 'JPEG', quality=95, optimize=True)
    plt.close(fig)

def setup_logging(log_file):
    """
    Setup logging for the application.

    Parameters:
        log_file (str): Path to the log file.
    """
    # Clear log file
    open(log_file, 'w').close()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )

def start_profiler():
    """
    Set up the profiler for the application.
    """
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    return profiler

def stop_profiler(profiler, output_file):
    """
    Stop the profiler and save the profiling data to a file.

    Parameters:
        profiler (cProfile.Profile): Profiler object.
        output_file (str): Path to save the profiling data.
    """
    profiler.disable()  # Stop profiling
    profiler.dump_stats(output_file)
    logging.info(f"Profiling data saved to {output_file}")
    # Optional: Launch SnakeViz visualization directly
    print("Run the following command to visualize the profiling data with SnakeViz:")
    print("snakeviz profile_data.prof")

def save_embeddings(embeddings, output_file):
    """
    Save the embeddings to a CSV file.

    Parameters:
        embeddings (torch.Tensor): Embeddings to save.
        output_file (str): Path to save the embeddings.
    """
    embeddings = pd.DataFrame(embeddings, columns=['spectrogram', 'embedding'])
    embeddings = pd.concat([embeddings.drop(columns=['embedding']), embeddings['embedding'].apply(pd.Series)], axis=1)
    embeddings.to_csv(output_file, index=False)
    logging.info(f"Embeddings saved to {output_file}")