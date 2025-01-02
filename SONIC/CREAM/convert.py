import torch
import torchaudio
import matplotlib.pyplot as plt

def waveform_to_image(waveform: torch.tensor, sr: int, n_mels: int = 128, hop_length: int = 512, n_fft: int = 2048, cmap: str = 'magma'):
    """
    Convert waveform to mel-spectrogram.
    Parameters:
        waveform (torch.Tensor): Input waveform.
        sr (int): Sampling rate of the waveform.
        n_mels (int, optional): Number of mel bands. Defaults to 128.
        hop_length (int, optional): Hop length for the STFT. Defaults to 512.
        n_fft (int, optional): Number of FFT points. Defaults to 2048.
    Returns:
        torch.Tensor: Mel-spectrogram.
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram).squeeze(0).cpu().numpy()
    epsilon = 1e-6
    mel_spectrogram_norm = (mel_spectrogram_db - mel_spectrogram_db.min() + epsilon) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + epsilon)

    cmap = plt.get_cmap(cmap)
    image = cmap(mel_spectrogram_norm)
    image = (image[:, :, :3] * 255)
    
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)