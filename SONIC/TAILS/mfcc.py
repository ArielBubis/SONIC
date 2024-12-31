import logging
from cv2 import transform
import torch
from torchaudio.transforms import MFCC
import torch.nn.functional as F
from tqdm.auto import tqdm
import SONIC.CREAM as CREAM

N_MFCC = 104  # Number of MFCC coefficients

def mfcc_embeddings(waveform):
    """
    Compute MFCC embeddings for the given waveform.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mfcc = MFCC(n_mfcc=N_MFCC, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}).to(device)
    waveform = waveform.to(device)
    mfcc_full = mfcc(waveform)
    embedding = mfcc_full.mean(dim=-1)
    return embedding

def get_embeddings(audio_dir: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = CREAM.dataset.init_dataset(audio_dir, batch_size=batch_size, transform=mfcc_embeddings)
    embeddings = []
    logging.info("Computing MFCC embeddings")
    logging.info(f"Using device: {device}")
    for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
        logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
        for audio_path, mfcc in zip(*batch):
            embeddings.append((audio_path, mfcc.cpu().numpy()))
            logging.info(f"Computed MFCC embedding for {audio_path}")

    return embeddings
