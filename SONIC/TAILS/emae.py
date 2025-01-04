import logging
from tqdm.auto import tqdm
from SONIC.CREAM import dataset
from SONIC.TAILS import embedder
from encodecmae import load_model
import torch
import warnings

class EncodecMAEEmbedder(embedder.Embedder):
    def __init__(self, batch_size=32):
        super().__init__(batch_size)
        try:
            self.model = load_model('mel256-ec-base_st', device='cuda:0')
        except Exception as e:
            logging.error(f"Error loading EncodecMAE model: {e}")
            print(f"Error loading EncodecMAE model: {e}")
            raise e
        logging.info("Initialized EncodecMAEEmbedder with EncodecMAE model")
        warnings.filterwarnings("ignore", category=FutureWarning)

    def embedding_fn(self, waveform):
        """
        Compute EncodecMAE embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform.
        Returns:
            torch.Tensor: Embedding generated by the EncodecMAE model.
        """
        waveform_np = waveform.cpu().numpy()
        with torch.no_grad():
            embedding = torch.tensor(self.model.extract_features_from_array(waveform_np))
        embedding = embedding.squeeze(0).mean(dim=0).squeeze()
        return embedding

    def get_embeddings(self, audio_dir):
        """
        Get embeddings for all audio files in the given directory.
        Parameters:
            audio_dir (str): Path to the directory containing audio files.
        Returns:
            list: List of tuples (audio_path, embedding).
        """
        dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=self.embedding_fn)
        embeddings = []
        
        logging.info("Computing EncodecMAE embeddings")
        logging.info(f"Using device: {self.device}")

        for i, batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
            logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
            for audio_path, embedding in zip(*batch):
                embeddings.append((audio_path, embedding.cpu().numpy()))
                logging.debug(f"Computed EncodecMAE embedding for {audio_path}")

        return embeddings
