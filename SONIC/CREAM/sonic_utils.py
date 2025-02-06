import cProfile
import logging
import numpy as np
import pandas as pd
import scipy.stats
from rs_metrics import ndcg, recall, hitrate, mrr
import subprocess
from tqdm.auto import tqdm
import multiprocessing
import os

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

def dict_to_pandas(d, key_col='user_id', val_col='item_id'):
    return (
        pd.DataFrame(d.items(), columns=[key_col, val_col])
            .explode(val_col)
            .reset_index(drop=True)
    )

def calc_metrics(test, pred, k=50):
    metrics = pd.DataFrame()
    metrics[f'NDCG@{k}'] = ndcg(test, pred, k=k, apply_mean=False)
    metrics[f'Recall@{k}'] = recall(test, pred, k=k, apply_mean=False)
    metrics[f'HitRate@{k}'] = hitrate(test, pred, k=k, apply_mean=False)
    metrics[f'MRR@{k}'] = mrr(test, pred, k=k, apply_mean=False)
    return metrics

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def safe_split(string, delimiter="_"):
    # Split the string on the delimiter
    split_result = string.split(delimiter, 1)  # Limit to 1 split for len 2
    # Pad with an empty string if the split result is shorter than 2
    return split_result if len(split_result) == 2 else [split_result[0], None]

def get_volume(file_path):
    try:
        cmd = [
            "ffmpeg", "-i", file_path, "-af", "volumedetect",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

        # Efficient extraction of the mean volume from FFmpeg output
        for line in result.stderr.splitlines():
            if "mean_volume" in line:
                volume_db = float(line.split(":")[-1].strip().replace(" dB", ""))
                return file_path, volume_db
    except Exception as e:
        # Log the error or use a custom handler if needed
        return file_path, None  # Mark as corrupt if FFmpeg fails

def scan_corrupt_audio(audio_dir: str, loudness_threshold: float) -> list:
    """
    Scan the given directory for corrupt audio files.

    Parameters:
        audio_dir (str): Path to the directory containing audio files.
        loudness_threshold (float): Minimum loudness threshold for valid audio files.

    Returns:
        list: List of corrupt audio files.
    """
    audio_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(audio_dir)
    for file in files if file.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus"))
]

    # Run volume check in parallel
    logging.info(f"Scanning {len(audio_files)} audio files...")
    logging.info(f"Using {multiprocessing.cpu_count() - 2} processes for parallel processing")
    logging.info(f"Loudness threshold: {loudness_threshold} dB")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        corrupt_files = list(tqdm(pool.imap(get_volume, audio_files), total=len(audio_files)))
    
    corrupt_files = list(filter(lambda x: x[1] is None or x[1] < loudness_threshold, corrupt_files))
    corrupt_files = [file for file, _ in corrupt_files]
    
    for file in corrupt_files:
        logging.warning(f"Corrupt audio file: {file}")

    return corrupt_files