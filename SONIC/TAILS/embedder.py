from abc import abstractmethod
import torch
import logging

class Embedder:
    def __init__(self, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        logging.info(f"Using device: {self.device}")
        logging.info(f"Batch size: {self.batch_size}")
    
    @abstractmethod
    def embedding_fn(self, waveform):
        """
        Compute embeddings for the given waveform.
        """
        pass

    @abstractmethod
    def get_embeddings(self, audio_dir):
        """
        Get the embeddings for the given input.
        """
        pass
        