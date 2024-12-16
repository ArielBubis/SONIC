import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fft import fft
import multiprocessing
from functools import partial
import concurrent.futures
import time
import psutil
import logging

import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('audio_visualization.log'),
        logging.StreamHandler()
    ]
)

def optimize_audio_visualization(audio_dir, output_dir, max_workers=None, file_extension='.mp3'):
    """
    Optimized audio visualization processing with multiple performance strategies.
    
    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated visualizations.
        max_workers (int, optional): Number of concurrent workers. 
                                     Defaults to number of CPU cores.
        file_extension (str, optional): File extension to process. Defaults to '.mp3'.
    """
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = max(1, psutil.cpu_count() - 1)  # Leave one core free
    
    # Create output directories
    visualization_dirs = {
        'mel_spectrograms': os.path.join(output_dir, "mel_spectrograms"),
        'chromagrams': os.path.join(output_dir, "chromagrams"),
        'fft_spectra': os.path.join(output_dir, "fft_spectra"),
        'tempograms': os.path.join(output_dir, "tempograms")
    }
    for dir_path in visualization_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Find all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(file_extension)]
    
    # Performance logging
    start_time = time.time()
    logging.info(f"Starting audio visualization for {len(audio_files)} files")
    logging.info(f"Using {max_workers} workers")
    
    # Prepare visualization parameters
    vis_params = {
        'mel_spectrogram': {
            'n_fft': 4096,
            'n_mels': 128,
            'gain_factor': 1.0,
            'vmin': -80,
            'vmax': 0
        }
    }
    
    # Parallel processing of audio files
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create visualization tasks
        visualization_tasks = [
            (generate_mel_spectrogram, (audio_dir, visualization_dirs['mel_spectrograms'], 
                                        vis_params['mel_spectrogram']['n_fft'], 
                                        vis_params['mel_spectrogram']['n_mels'], 
                                        vis_params['mel_spectrogram']['gain_factor'], 
                                        vis_params['mel_spectrogram']['vmin'], 
                                        vis_params['mel_spectrogram']['vmax'])),
            (generate_chromagram, (audio_dir, visualization_dirs['chromagrams'])),
            (generate_fft_spectrum, (audio_dir, visualization_dirs['fft_spectra'])),
            (generate_tempogram, (audio_dir, visualization_dirs['tempograms']))
        ]
        
        # Submit visualization tasks
        futures = []
        for func, args in visualization_tasks:
            futures.append(executor.submit(func, *args))
        
        # Wait for all tasks to complete and handle any exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in visualization task: {e}")
    
    # Performance reporting
    end_time = time.time()
    logging.info(f"Total Processing Time: {end_time - start_time:.2f} seconds")
    logging.info(f"Number of files processed: {len(audio_files)}")
    logging.info(f"Workers used: {max_workers}")





def generate_mel_spectrogram(audio_dir, output_dir, n_fft, n_mels, gain_factor=1.0, vmin=-80, vmax=0):
    """
    Generate and save Mel spectrograms for audio files in a directory.

    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated spectrograms.
        n_fft (int): FFT window size.
        n_mels (int): Number of Mel bands.
        gain_factor (float): Factor to amplify the Mel spectrogram.
        vmin (float): Minimum dB value for the spectrogram color scale.
        vmax (float): Maximum dB value for the spectrogram color scale.
    """
    # mel_dir = os.path.join(output_dir, "mel_spectrograms")
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    for filename in audio_files:
        filepath = os.path.join(audio_dir, filename)
        try:
            y, sr = librosa.load(filepath, sr=None)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        hop_length = utils.calculate_dynamic_hop_length(n_fft, sr)
        hop_length = max(hop_length, n_mels)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        mel_spectrogram *= gain_factor
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        fig, ax = plt.subplots(figsize=(50, 10))
        
        librosa.display.specshow(
            mel_spectrogram_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=ax, cmap="magma",
            vmin=vmin, vmax=vmax
        )
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding and margins
        output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        utils.save_image_with_cv2(fig, output_filepath)
        print(f"Saved Mel spectrogram: {output_filepath}")


def generate_chromagram(audio_dir, output_dir):
    """
    Generate and save chromagrams for audio files in a directory.

    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated chromagrams.
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    for filename in audio_files:
        filepath = os.path.join(audio_dir, filename)
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap="coolwarm", ax=ax)
        ax.set_title("Chromagram")
        output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        utils.save_image_with_cv2(fig, output_filepath)
        print(f"Saved Chromagram: {output_filepath}")


def generate_fft_spectrum(audio_dir, output_dir):
    """
    Generate and save FFT spectra for audio files in a directory.

    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save FFT spectra.
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    for filename in audio_files:
        filepath = os.path.join(audio_dir, filename)
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        X = fft(y)
        f = np.linspace(0, sr, len(X))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(f[:len(f)//2], np.abs(X)[:len(f)//2])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('FFT Spectrum')
        output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        utils.save_image_with_cv2(fig, output_filepath)
        print(f"Saved FFT Spectrum: {output_filepath}")


def generate_tempogram(audio_dir, output_dir):
    """
    Generate and save tempograms for audio files in a directory.

    A tempogram visualizes rhythmic patterns and tempo variations in audio signals.

    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated tempograms.
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    for filename in audio_files:
        filepath = os.path.join(audio_dir, filename)
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Compute tempogram: local autocorrelation of onset strength
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.specshow(
            tempogram, 
            x_axis='time', 
            y_axis='tempo', 
            cmap='viridis', 
            ax=ax
        )
        ax.set_title('Tempogram')
        ax.set_xlabel('Time')
        ax.set_ylabel('Tempo (BPM)')
        
        output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_tempogram.png")
        utils.save_image_with_cv2(fig, output_filepath)
        print(f"Saved Tempogram: {output_filepath}")

# # Update the main block to include new visualization functions
# if __name__ == "__main__":
#     audio_dir = 'music4all/music4all/audios'  # Replace with your directory
#     output_dir = 'audio_visualizations'
#     n_fft = 4096  # FFT window size
#     n_mels = 128  # Number of Mel bands
#     gain_factor = 1.0  # Adjust gain if needed
#     vmin = -80  # Minimum dB value for the color scale
#     vmax = 0  # Maximum dB value for the color scale

#     # Generate different visualizations
#     generate_mel_spectrogram(audio_dir, output_dir, n_fft, n_mels, gain_factor, vmin, vmax)
#     generate_chromagram(audio_dir, output_dir)
#     generate_fft_spectrum(audio_dir, output_dir)
#     generate_tempogram(audio_dir, output_dir)
