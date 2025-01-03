from tqdm.auto import tqdm
import logging
from TAILS import embedder
from CREAM import dataset
from musicfm.model.musicfm_25hz import MusicFM25Hz
import torch
import torch.nn as nn

class MusicFMEmbedder(embedder.Embedder):
    def __init__(self, batch_size=32):
        super().__init__(batch_size)
        self.model = MusicFM25Hz(is_flash=False).to(self.device)
        self.projection = nn.Linear(500, 750).to(self.device)

    def embedding_fn(self, waveform):
        """
        Compute MusicFM embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform of shape (batch_size, num_samples)
        Returns:
            torch.Tensor: MusicFM embedding of size 750.
        """
        
        # Ensure the input is on the correct device
        waveform = waveform.to(self.device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Get embeddings from the model
        emb = self.model.get_latent(waveform, layer_ix=12).mean(-1)


        self.model.eval()
        # Debug: Check embedding shape
        logging.debug(f"Raw embedding shape: {emb.shape}")
        emb = self.projection(emb)
        
        return emb.cpu().detach().numpy().flatten()
        

    def get_embeddings(self, audio_dir):
        dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=self.embedding_fn)
        embeddings = []
        
        logging.info("Computing MusicFM embeddings")
        logging.info(f"Using device: {self.device}")
        
        for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
            for audio_path, embedding in zip(*batch):
                embeddings.append((audio_path, embedding))
                logging.debug(f"Computed MusicFM embedding for {audio_path}")
        
        return embeddings