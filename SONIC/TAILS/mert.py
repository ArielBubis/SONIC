from SONIC.TAILS import embedder
from SONIC.TAILS import encoder
from SONIC.CREAM import dataset
from tqdm.auto import tqdm
import logging
import torch


class MERTEmbedder(embedder.Embedder):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.mert = encoder.get_encoder("m-a-p/MERT-v1-330M").to(self.device)

    def embedding_fn(self, waveform):
        """
        Compute MERT embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform.
        Returns:
            torch.Tensor: MERT embedding.
        """
        waveform = waveform.to(self.device)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            return self.mert(waveform).last_hidden_state.mean(-2).cpu().squeeze().numpy()
    
    def get_embeddings(self, audio_dir):
        dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=self.embedding_fn)
        embeddings = []
        self.mert.eval()
        logging.info("Computing MERT embeddings")
        for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
            logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
            for audio_path, embedding in zip(*batch):
                embeddings.append((audio_path, embedding.squeeze()))
                logging.info(f"Computed MERT embedding for {audio_path}")
        
        return embeddings