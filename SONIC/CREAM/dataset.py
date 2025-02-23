from typing import Callable
import torch
from torchaudio import load
from torchaudio.transforms import Resample
import torch.nn.functional as F
import os

def custom_collate(batch):
    """
    Custom collate function to flatten nested batches.

    Args:
        batch (list): List of tuples containing audio file names and waveforms.

    Returns:
        tuple: List of audio file names and a batch tensor of waveforms.
    """
    audio_files, waveforms = zip(*batch)

    print([waveform.shape for waveform in waveforms])

    # Ensure all waveforms are 1D and have consistent lengths
    max_length = max(waveform.size(0) for waveform in waveforms)
    waveforms = [
        F.pad(waveform, (0, max_length - waveform.size(-1))) for waveform in waveforms
    ]

    waveforms_batch = torch.stack(waveforms)
    return list(audio_files), waveforms_batch

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading audio data.

    Args:
        audio_dir (str): Directory containing audio files.
        transform (Callable, optional): Optional transform to be applied on a sample.

    Attributes:
        audio_dir (str): Directory containing audio files.
        audio_files (list): List of audio file names.
        transform (Callable): Optional transform to be applied on a sample.
    """
    AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.opus']
    AUDIO_SR = 16_000 
    def __init__(self, audio_dir, transform: Callable=None):
        self.audio_dir = audio_dir
        self.audio_files = os.listdir(audio_dir)
        self.audio_files = [f for f in self.audio_files if os.path.splitext(f)[1] in AudioDataset.AUDIO_FORMATS]
        self.transform = transform
        

    def __len__(self):
        """
        Return the total number of audio files.
        """
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Audio file name and waveform tensor.
        """

        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        waveform, sr = load(audio_path)
        resample = Resample(sr, AudioDataset.AUDIO_SR)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True).squeeze(0)
        waveform = resample(waveform)
        
        if self.transform:
            waveform = self.transform(waveform)

        return audio_file, waveform

def init_dataset(audio_dir, batch_size=32, transform=None):
    """
    Initialize the dataset for the given audio directory.

    Args:
        audio_dir (str): Directory containing audio files.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        transform (Callable, optional): Optional transform to be applied on a sample.

    Returns:
        DataLoader: DataLoader for the audio dataset.
    """
    dataset = AudioDataset(audio_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
