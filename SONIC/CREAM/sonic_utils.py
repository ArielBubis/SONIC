import cProfile
import logging
import numpy as np
import pandas as pd
import scipy.stats
from rs_metrics import ndcg, recall, hitrate, mrr

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