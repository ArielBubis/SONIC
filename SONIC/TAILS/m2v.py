from tqdm.auto import tqdm
from SONIC.TAILS import embedder, encoder
from SONIC.CREAM import dataset
import logging
import torch

class M2VEmbedder(embedder.Embedder):
    """
    M2VEmbedder class for computing Music2Vec (M2V) embeddings.
    Args:
        batch_size (int): Batch size for processing audio files.
    """
    def __init__(self, batch_size):
        super().__init__(batch_size)
        # Initialize the Music2Vec encoder
        self.m2v = encoder.get_encoder("m-a-p/music2vec-v1").to(self.device)
    
    def embedding_fn(self, waveform):
        """
        Compute M2V embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform.
        Returns:
            torch.Tensor: M2V embedding.
        """
        waveform = waveform.to(self.device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            return self.m2v(waveform).last_hidden_state.mean(-2).cpu().squeeze().numpy()
    
    def get_embeddings(self, audio_dir):
        dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=self.embedding_fn)
        embeddings = []
        self.m2v.eval()
        logging.info("Computing M2V embeddings")
        for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
            logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
            for audio_path, embedding in zip(*batch):
                embeddings.append((audio_path, embedding.squeeze()))
                logging.info(f"Computed M2V embedding for {audio_path}")
        
        return embeddings
