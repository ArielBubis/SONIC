# import os
# import concurrent.futures
# import time
# import librosa
# import psutil
# import logging
# import tqdm
# import gc
# # from . import io
# # from .sonic_utils im
# import torch
# from torchvision.transforms.functional import resize
# import numpy as np

# def spectrogram_to_audio(spectrogram, output_path, sample_rate=44_100):
#     """
#     Convert a mel-spectrogram to audio.
#     Parameters:
#         spectrogram (torch.Tensor): Mel-spectrogram.
#         output_path (str): Path to save the audio file.
#         sample_rate (int, optional): Sampling rate for the audio file. Defaults to 22050.

#     Returns:
#         np.ndarray: Audio signal.
#         int: Sample rate.
#     """

#     # Convert to decibels
#     mel_spec_amplitude = image_to_db(spectrogram).cpu().numpy()

#     print("Spectrogram Stats:")
#     print(f"Min: {torch.min(spectrogram)}")
#     print(f"Max: {torch.max(spectrogram)}")
#     print(f"Mean: {torch.mean(spectrogram)}")

#     print("Decibel Stats:")
#     print(f"Min: {np.min(mel_spec_amplitude)}")
#     print(f"Max: {np.max(mel_spec_amplitude)}")
#     print(f"Mean: {np.mean(mel_spec_amplitude)}")


#     hop_length = calculate_dynamic_hop_length(sample_rate)
#     n_fft = 2048
#     n_mels = 128

#     mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
#     linear_spec = np.dot(np.linalg.pinv(mel_filter), mel_spec_amplitude)

#     print("Linear Spectrogram Stats:")
#     print(f"Min: {np.min(linear_spec)}")
#     print(f"Max: {np.max(linear_spec)}")
#     print(f"Mean: {np.mean(linear_spec)}")

#     # Ensure non-negative values (numerical errors may cause tiny negatives)
#     linear_spec = np.maximum(0, linear_spec)
#     if len(linear_spec.shape) > 2:
#         linear_spec = linear_spec.squeeze(axis=1)
#     print(f"mel_spec_amplitude shape: {mel_spec_amplitude.shape}")
#     print(f"mel_filter shape: {mel_filter.shape}")
#     print(f"linear_spec shape: {linear_spec.shape}")

#     # Reconstruct waveform using Griffin-Lim
#     audio = librosa.griffinlim(linear_spec, n_iter=32, hop_length=hop_length, n_fft=n_fft)

    
#     # y = librosa.feature.inverse.mel_to_audio(mel_spectrogram, sr=sample_rate)
#     io.save_audio(audio, output_path, sample_rate)
#     return audio, sample_rate

# def audio_to_spectograms(audio_dir, output_dir, fig_size, max_workers=None, file_extension='.mp3'):
#     """
#     Convert audio files diretory to mel spectrograms, chromagrams, and tempograms.
#     Parameters:
#         audio_dir (str): Directory containing audio files.
#         output_dir (str): Directory to save generated spectrograms.
#         fig_size (tuple): Size of the figure.
#         max_workers (int, optional): Number of concurrent workers. Defaults to number of CPU cores.
#         file_extension (str, optional): File extension to process. Defaults to '.mp3'.
#         log_file (str, optional): Log file path. Defaults to 'audio_visualization.log'.    
#     """

#     # Determine optimal number of workers
#     if max_workers is None:
#         max_workers = max(1, psutil.cpu_count() - 2)  # Leave one core free
       
#     # Create output directories
#     visualization_dirs = {
#         'mel_spectrograms': os.path.join(output_dir, "mel_spectrograms"),
#         'chromagrams': os.path.join(output_dir, "chromagrams"),
#         'tempograms': os.path.join(output_dir, "tempograms")
#     }
#     for dir_path in visualization_dirs.values():
#         os.makedirs(dir_path, exist_ok=True)
   
#     # Find all audio files
#     audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(file_extension)]

#     start_time = time.time()
#     logging.info(f"Starting audio visualization for {len(audio_files)} files")
#     logging.info(f"Using {max_workers} workers")

#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         tasks = []
#         for idx, audio_file in enumerate(audio_files):
#             logging.info(f"Submitting task {idx + 1}/{len(audio_files)}: {audio_file}")
#             tasks.append(
#                 executor.submit(
#                     process_audio_file,
#                     os.path.join(audio_dir, audio_file),
#                     visualization_dirs,
#                     fig_size
#                 )
#             )

#         for idx, future in enumerate(tqdm.tqdm(concurrent.futures.as_completed(tasks), total=len(audio_files), desc="Processing files")):
#             try:
#                 future.result()
#                 logging.info(f"Task {idx + 1}/{len(audio_files)} completed successfully.")
#             except Exception as e:
#                 logging.warning(f"Error in task {idx + 1}/{len(audio_files)}: {e}")

#     end_time = time.time()
#     logging.info(f"Total Processing Time: {end_time - start_time:.2f} seconds")
#     logging.info(f"Number of files processed: {len(audio_files)}")
#     logging.info(f"Workers used: {max_workers}")

# def process_audio_file(filepath, visualization_dirs, fig_size):
#     try:
#         logging.info(f"Processing file: {filepath}")
#         generate_spectogram(filepath, visualization_dirs['mel_spectrograms'], fig_size, 'mel')
#         logging.info(f"Generated mel spectrogram for {filepath}")
#         generate_spectogram(filepath, visualization_dirs['chromagrams'], fig_size, 'chroma')
#         logging.info(f"Generated chromagram for {filepath}")
#         generate_spectogram(filepath, visualization_dirs['tempograms'], fig_size, 'tempogram')
#         logging.info(f"Generated tempogram for {filepath}")
#         logging.info(f"Finished processing file: {filepath}")
#     except Exception as e:
#         logging.error(f"Error processing file: {filepath} - {e}")
#     finally:
#         gc.collect()  # Explicitly call garbage collector
