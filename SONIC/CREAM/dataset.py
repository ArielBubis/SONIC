from typing import Callable
import torch
from torchaudio import load
from torchaudio.transforms import Resample
import torch.nn.functional as F
import os

def custom_collate(batch):
    """
    Custom collate function to flatten nested batches.
    """
    audio_files, waveforms = zip(*batch)

    print([waveform.shape for waveform in waveforms])

    # Ensure all waveforms are 1D and have consistent lengths
    max_length = max(waveform.size(0) for waveform in waveforms)
    waveforms = [
        F.pad(waveform, (0, max_length - waveform.size(-1))) for waveform in waveforms
    ]

    # Stack waveforms into a batch tensor
    waveforms_batch = torch.stack(waveforms)
    return list(audio_files), waveforms_batch

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading audio data
    """
    AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', 'opus']
    AUDIO_SR = 16_000 # Audio sample rate in Hz (16 kHz)
    def __init__(self, audio_dir, transform: Callable=None):
        self.audio_dir = audio_dir
        self.audio_files = os.listdir(audio_dir)
        self.audio_files = [f for f in self.audio_files if os.path.splitext(f)[1] in AudioDataset.AUDIO_FORMATS]
        self.transform = transform
        

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
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
    """
    dataset = AudioDataset(audio_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
