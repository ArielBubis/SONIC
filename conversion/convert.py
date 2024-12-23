import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import time
import psutil
import logging
import tqdm
import conversion.utils as utils
import gc
from spectograms import generate_spectogram

def audio_to_spectograms(audio_dir, output_dir, fig_size, max_workers=None, file_extension='.mp3', log_file='audio_visualization.log'):
    """
    Convert audio files diretory to mel spectrograms, chromagrams, and tempograms.
    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated spectrograms.
        fig_size (tuple): Size of the figure.
        max_workers (int, optional): Number of concurrent workers. Defaults to number of CPU cores.
        file_extension (str, optional): File extension to process. Defaults to '.mp3'.
        log_file (str, optional): Log file path. Defaults to 'audio_visualization.log'.    
    """

    # Setup logging
    open(log_file, 'w').close()
    plt.switch_backend('Agg')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )

    # Determine optimal number of workers
    if max_workers is None:
        max_workers = max(1, psutil.cpu_count() - 2)  # Leave one core free
       
    # Create output directories
    visualization_dirs = {
        'mel_spectrograms': os.path.join(output_dir, "mel_spectrograms"),
        'chromagrams': os.path.join(output_dir, "chromagrams"),
        'tempograms': os.path.join(output_dir, "tempograms")
    }
    for dir_path in visualization_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
   
    # Find all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(file_extension)]

    start_time = time.time()
    logging.info(f"Starting audio visualization for {len(audio_files)} files")
    logging.info(f"Using {max_workers} workers")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for idx, audio_file in enumerate(audio_files):
            logging.info(f"Submitting task {idx + 1}/{len(audio_files)}: {audio_file}")
            tasks.append(
                executor.submit(
                    process_audio_file,
                    os.path.join(audio_dir, audio_file),
                    visualization_dirs,
                    fig_size
                )
            )

        for idx, future in enumerate(tqdm.tqdm(concurrent.futures.as_completed(tasks), total=len(audio_files), desc="Processing files")):
            try:
                future.result()
                logging.info(f"Task {idx + 1}/{len(audio_files)} completed successfully.")
            except Exception as e:
                logging.warning(f"Error in task {idx + 1}/{len(audio_files)}: {e}")

    end_time = time.time()
    logging.info(f"Total Processing Time: {end_time - start_time:.2f} seconds")
    logging.info(f"Number of files processed: {len(audio_files)}")
    logging.info(f"Workers used: {max_workers}")

def process_audio_file(filepath, visualization_dirs, fig_size):
    try:
        logging.info(f"Processing file: {filepath}")
        generate_spectogram(filepath, visualization_dirs['mel_spectrograms'], fig_size, 'mel')
        logging.info(f"Generated mel spectrogram for {filepath}")
        generate_spectogram(filepath, visualization_dirs['chromagrams'], fig_size, 'chroma')
        logging.info(f"Generated chromagram for {filepath}")
        generate_spectogram(filepath, visualization_dirs['tempograms'], fig_size, 'tempogram')
        logging.info(f"Generated tempogram for {filepath}")
        logging.info(f"Finished processing file: {filepath}")
    except Exception as e:
        logging.error(f"Error processing file: {filepath} - {e}")
    finally:
        gc.collect()  # Explicitly call garbage collector
