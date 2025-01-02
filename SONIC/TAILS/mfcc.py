from functools import partial
import logging
import torch
from torchaudio.transforms import MFCC
from tqdm.auto import tqdm
from SONIC.CREAM import dataset
from functools import partial
from SONIC.TAILS import embedder

N_MFCC = 104  # Number of MFCC coefficients in the original implementation

class MFCCEmbedder(embedder.Embedder):
    def __init__(self, batch_size=32, n_mfcc=N_MFCC):
        super().__init__(batch_size)
        self.mfcc = MFCC(n_mfcc=n_mfcc, melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}).to(self.device)
        logging.info(f"Number of MFCC coefficients: {n_mfcc}")

    def embedding_fn(self, waveform):
        """
        Compute MFCC embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform.
        Returns:
            torch.Tensor: MFCC embedding.
        """
        waveform = waveform.to(self.device)
        mfcc_full = self.mfcc(waveform)
        embedding = mfcc_full.mean(dim=-1)
        if embedding.dim() == 2:
            embedding = embedding.squeeze(0)
        return embedding

    def get_embeddings(self, audio_dir):
        dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=self.embedding_fn)
        embeddings = []
        logging.info("Computing MFCC embeddings")
        logging.info(f"Using device: {self.device}")
        for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
            logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
            for audio_path, mfcc in zip(*batch):
                embeddings.append((audio_path, mfcc.cpu().numpy()))
                logging.info(f"Computed MFCC embedding for {audio_path}")
        
        return embeddings
