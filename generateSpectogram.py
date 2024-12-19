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
import utils

open('audio_visualization.log', 'w').close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('audio_visualization.log'),
        # logging.StreamHandler()
    ]
)

def batchify(lst, batch_size):
    """
    Helper function to divide the audio file list into smaller batches.
    
    Parameters:
        lst (list): The list to be divided.
        batch_size (int): Number of elements in each batch.
    
    Returns:
        list: A list of batches.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def audio_visualization(audio_dir, output_dir, fig_size, max_workers=None, file_extension='.mp3'):    
    """
    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated visualizations.
        fig_size (tuple): Size of the figure.
        max_workers (int, optional): Number of concurrent workers. Defaults to number of CPU cores.
        batch_size (int, optional): Number of files to process per batch. Defaults to 10.
        file_extension (str, optional): File extension to process. Defaults to '.mp3'.    
    """
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = max(1, psutil.cpu_count() - 1)  # Leave one core free
       
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
    batches = list(batchify(audio_files, 3000))

    start_time = time.time()
    for batch_num, batch in enumerate(batches, start=1):
        print(f"Processing batch {batch_num}/{len(batches)}")
        logging.info(f"Processing batch {batch_num}/{len(batches)}")
        logging.info(f"Starting audio visualization for {len(audio_files)} files")
        logging.info(f"Using {max_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for each file
            tasks = [
                executor.submit(
                    process_audio_file,
                    os.path.join(audio_dir, audio_file),
                    visualization_dirs,
                    fig_size
                )
                for audio_file in batch
            ]

            # Wait for all tasks and log progress
            for future in tqdm.tqdm(concurrent.futures.as_completed(tasks), total=len(batch_num), desc="Processing files"):
                try:
                    future.result()
                    # logging.info("File processed successfully")
                except Exception as e:
                    tqdm.error(f"Error in processing file: {e}")

    # Performance reporting
    end_time = time.time()
    logging.info(f"Total Processing Time: {end_time - start_time:.2f} seconds")
    logging.info(f"Number of files processed: {len(audio_files)}")
    logging.info(f"Workers used: {max_workers}")


def process_audio_file(filepath, visualization_dirs, fig_size):
    """
    Process a audio file to generate all spectrogram types.
    """
    vis_params = {
        'mel_spectrogram': {
            'n_fft': 4096,
            'n_mels': 128,
            'gain_factor': 1.0,
            'vmin': -80,
            'vmax': 0
        }
    }
    generate_visualization(filepath, visualization_dirs['mel_spectrograms'], fig_size, 'mel',
                                    n_fft=vis_params['mel_spectrogram']['n_fft'],
                                    n_mels=vis_params['mel_spectrogram']['n_mels'],
                                    gain_factor=vis_params['mel_spectrogram']['gain_factor'],
                                    vmin=vis_params['mel_spectrogram']['vmin'],
                                    vmax=vis_params['mel_spectrogram']['vmax'])
    generate_visualization(filepath, visualization_dirs['chromagrams'], fig_size, 'chroma')
    generate_visualization( filepath, visualization_dirs['tempograms'], fig_size, 'tempogram')        

def generate_visualization(audio_dir, output_dir, fig_size, spectrogram_type, **kwargs):
    """
    Generate and save spectrograms for audio files in a directory.

    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated spectrograms.
        fig_size (tuple): Size of the figure.
        spectrogram_type (str): Type of spectrogram to generate ('mel', 'chroma', 'tempogram').
        kwargs: Additional parameters for specific spectrogram types.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_dir))[0]}.jpeg")
    if os.path.exists(output_filepath):
        logging.info(f"File already exists: {output_filepath}")
        return
    try:
        y, sr = librosa.load(audio_dir, sr=None, mono=True,duration=30)
    except Exception as e:
        print(f"Error loading {audio_dir}: {e}")
    if y is None or len(y) == 0:
        logging.warning(f"Empty or invalid audio signal for file: {audio_dir}")
        return
    
    n_fft = kwargs.get('n_fft', 2048)
    n_mels = kwargs.get('n_mels', 128)
    gain_factor = kwargs.get('gain_factor', 1.0)
    vmin = kwargs.get('vmin', -80)
    vmax = kwargs.get('vmax', 0)
    hop_length = utils.calculate_dynamic_hop_length(n_fft, sr)
    hop_length = max(hop_length, n_mels)

    # Generate spectrogram based on type and save
    if spectrogram_type == 'mel':
        # Compute mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        spectrogram *= gain_factor
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        display_func = librosa.display.specshow
        display_kwargs = {
            'x_axis': 'time', 'y_axis': 'mel', 'cmap': 'magma', 'vmin': vmin, 'vmax': vmax
        }
    elif spectrogram_type == 'chroma':
        # Compute chromagram
        spectrogram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        display_func = librosa.display.specshow
        display_kwargs = {'x_axis': 'time', 'y_axis': 'chroma', 'cmap': 'coolwarm'}

    elif spectrogram_type == 'tempogram':
        # Compute tempogram
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectrogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        display_func = librosa.display.specshow
        display_kwargs = {'x_axis': 'time', 'y_axis': 'tempo', 'cmap': 'viridis'}
    else:
        print(f"Unknown spectrogram type: {spectrogram_type}")
        return

    fig, ax = plt.subplots(figsize=fig_size)
    display_func(spectrogram, ax=ax, **display_kwargs)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding and margins

    utils.save_image_with_cv2(fig, output_filepath)
    logging.info(f"Saved {spectrogram_type.capitalize()} spectrogram: {output_filepath}")

    # print(f"Saved {spectrogram_type.capitalize()} spectrogram: {output_filepath}")