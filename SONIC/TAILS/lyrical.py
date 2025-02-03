from scipy.__config__ import show
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm.auto import tqdm
from glob import glob
import os



class LyricalEmbedder:
    """
    Class to get embeddings for lyrics.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to(self.device)

    def get_lyrics_embeddings(self, lyrics: str) -> np.ndarray:
        """
        Get the embeddings of the given lyrics.
        Parameters:
            lyrics (str): Lyrics of the song.
        Returns:
            np.ndarray: Lyrics embeddings.
        """
        if lyrics == "":
            lyrics = "[Instrumental_Track]"
        return self.text_model.encode(lyrics, show_progress_bar=False)
    
    def get_embeddings(self, lyrics_dir: str):
        """
        Get embeddings for all lyrics files in the given directory.
        Parameters:
            lyrics_dir (str): Path to the directory containing lyrics files.
        Returns:
            list: List of tuples (lyrics_path, embedding).
        """
        embeddings = []
        files = glob(f"{lyrics_dir}/*.txt", recursive=True)
        for file in tqdm(files):
            with open(file, "r") as f:
                lyrics = f.read()
            embeddings.append((file, self.get_lyrics_embeddings(lyrics)))
        return embeddings