import matplotlib.pyplot as plt
from PIL import Image
import cProfile
import logging
import torch
from torchvision import transforms
import os

class SpectrogramDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading spectrograms.
    """
    def __init__(self, spectograms_dir):
        self.spectograms_dir = spectograms_dir
        self.spectograms = os.listdir(spectograms_dir)

    def __len__(self):
        return len(self.spectograms)

    def __getitem__(self, idx):
        spectogram_path = os.path.join(self.spectograms_dir, self.spectograms[idx])
        spectogram = Image.open(spectogram_path)
        spectogram = transforms.ToTensor()(spectogram)
        return spectogram

def init_dataset(spectograms_dir, batch_size=32):
    """
    Initialize the dataset for the given spectrogram directory.

    Parameters:
        spectograms_dir (str): Path to the directory containing spectrograms.
        batch_size (int): Batch size for the dataset.

    Returns:
        torch.utils.data.DataLoader: DataLoader object.
    """
    # Load the dataset
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
