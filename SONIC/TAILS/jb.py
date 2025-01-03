from jukemirlib import extract
from SONIC.TAILS import embedder
from SONIC.CREAM import dataset
from tqdm.auto import tqdm

N_LAYERS = 36  # Number of layers in the Jukebox model
duration = 30  # Duration of the audio clip for embedding
use_fp16 = True  # Use FP16 for faster inference

class JukeboxEmbedder(embedder.Embedder):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def embedding_fn(self, audio_path):
        """
        Compute Jukebox embeddings for the given audio file.
        Parameters:
            audio_path (str): Path to the audio file.
        Returns:
            torch.Tensor: Jukebox embedding.
        """
        return extract(fpath=audio_path, fp16=use_fp16, layers=[N_LAYERS], duration=duration)[N_LAYERS].mean(axis=1)

    def get_embeddings(self, audio_dir):
        dataloader = dataset.init_dataset(audio_dir, batch_size=self.batch_size, transform=None)
        embeddings = []
        for audio_path in tqdm(dataloader, desc="Extracting embeddings"):
            embeddings.append((audio_path, self.embedding_fn(audio_path)))
        return embeddings