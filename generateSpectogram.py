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
import gc

open('audio_visualization.log', 'w').close()
plt.switch_backend('Agg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('audio_visualization.log'),
    ]
)

def audio_visualization(audio_dir, output_dir, fig_size, max_workers=None, file_extension='.mp3'):    
    """
    Parameters:
        audio_dir (str): Directory containing audio files.
        output_dir (str): Directory to save generated visualizations.
        fig_size (tuple): Size of the figure.
        max_workers (int, optional): Number of concurrent workers. Defaults to number of CPU cores.
        file_extension (str, optional): File extension to process. Defaults to '.mp3'.    
    """
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
        generate_visualization(filepath, visualization_dirs['mel_spectrograms'], fig_size, 'mel')
        logging.info(f"Generated mel spectrogram for {filepath}")
        generate_visualization(filepath, visualization_dirs['chromagrams'], fig_size, 'chroma')
        logging.info(f"Generated chromagram for {filepath}")
        generate_visualization(filepath, visualization_dirs['tempograms'], fig_size, 'tempogram')
        logging.info(f"Generated tempogram for {filepath}")
        logging.info(f"Finished processing file: {filepath}")
    except Exception as e:
        logging.error(f"Error processing file: {filepath} - {e}")
    finally:
        gc.collect()  # Explicitly call garbage collector

def generate_visualization(audio_dir, output_dir, fig_size, spectrogram_type):
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_dir))[0]}.jpeg")

    if os.path.exists(output_filepath):
        logging.info(f"File already exists: {output_filepath}")
        return
    
    try:
        y, sr = librosa.load(audio_dir, sr=None, mono=True, duration=30)
        logging.info(f"Loaded audio file: {audio_dir} with sampling rate {sr}")
    except Exception as e:
        logging.error(f"Error loading {audio_dir}: {e}")
        return
    
    if y is None or len(y) == 0:
        logging.warning(f"Empty or invalid audio signal for file: {audio_dir}")
        return
    
    hop_length = utils.calculate_dynamic_hop_length(sr)
    hop_length = max(hop_length, 128)

    try:
        if spectrogram_type == 'mel':
            logging.info(f"Generating mel spectrogram for {audio_dir}")
            generate_mel_spectrogram(y, sr, hop_length, output_filepath, fig_size)
        elif spectrogram_type == 'chroma':
            logging.info(f"Generating chromagram for {audio_dir}")
            generate_chromagram(y, sr, hop_length, output_filepath, fig_size)
        elif spectrogram_type == 'tempogram':
            logging.info(f"Generating tempogram for {audio_dir}")
            generate_tempogram(y, sr, hop_length, output_filepath, fig_size)
        else:
            logging.error(f"Unknown spectrogram type: {spectrogram_type}")
    except Exception as e:
        logging.error(f"Error generating {spectrogram_type} spectrogram for file {audio_dir}: {e}")

def generate_mel_spectrogram(y, sr, hop_length, output_filepath, fig_size):
    try:
        n_fft = 2048
        n_mels = 128
        gain_factor = 1.0
        vmin = -80
        vmax = 0

        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        spectrogram *= gain_factor
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        display_func = librosa.display.specshow
        display_kwargs = {
            'x_axis': 'time', 'y_axis': 'mel', 'cmap': 'magma', 'vmin': vmin, 'vmax': vmax
        }
        save_spectrogram(spectrogram, display_func, display_kwargs, output_filepath, fig_size, 'mel-spectrogram')
        logging.info(f"Saved mel spectrogram to {output_filepath}")
    except Exception as e:
        logging.error(f"Error generating mel spectrogram: {e}")

def generate_chromagram(y, sr, hop_length, output_filepath, fig_size):
    try:
        spectrogram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        display_func = librosa.display.specshow
        display_kwargs = {'x_axis': 'time', 'y_axis': 'chroma', 'cmap': 'coolwarm'}
        save_spectrogram(spectrogram, display_func, display_kwargs, output_filepath, fig_size, 'chroma')
        logging.info(f"Saved chromagram to {output_filepath}")
    except Exception as e:
        logging.error(f"Error generating chromagram: {e}")

def generate_tempogram(y, sr, hop_length, output_filepath, fig_size):
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectrogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        display_func = librosa.display.specshow
        display_kwargs = {'x_axis': 'time', 'y_axis': 'tempo', 'cmap': 'viridis'}
        save_spectrogram(spectrogram, display_func, display_kwargs, output_filepath, fig_size, 'tempogram')
        logging.info(f"Saved tempogram to {output_filepath}")
    except Exception as e:
        logging.error(f"Error generating tempogram: {e}")

def save_spectrogram(spectrogram, display_func, display_kwargs, output_filepath, fig_size,spectrogram_type):
    try:
        fig, ax = plt.subplots(figsize=fig_size)
        display_func(spectrogram, ax=ax, **display_kwargs)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding and margins
        utils.save_image_with_pillow(fig, output_filepath)
        logging.info(f"Saved {spectrogram_type.capitalize()} spectrogram: {output_filepath}")
        # print(f"Saved {spectrogram_type.capitalize()} spectrogram: {output_filepath}")
        plt.close(fig)
        logging.info(f"Spectrogram saved at {output_filepath}")
    except Exception as e:
        logging.error(f"Error saving spectrogram: {e}")
