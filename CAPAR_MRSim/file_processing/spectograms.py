import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

def generate_spectogram(audio_dir, output_dir, fig_size, spectrogram_type):
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