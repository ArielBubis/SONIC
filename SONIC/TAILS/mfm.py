HOME_PATH = "./" # path where you cloned musicfm


import os
import sys
import torch


sys.path.append(HOME_PATH)
from SONIC.TAILS import embedder
from musicfm.model.musicfm_25hz import MusicFM25Hz

class MusicFMEmbedder(embedder.Embedder):
    def __init__(self, batch_size=32):
        super().__init__(batch_size)
        self.model = MusicFM25Hz().to(self.device)
        self.model.eval()

    def embedding_fn(self, waveform):
        """
        Compute MusicFM embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform.
        Returns:
            torch.Tensor: MusicFM embedding.
        """
        musicfm = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(HOME_PATH, "musicfm", "data", "msd_stats.json"),
            model_path=os.path.join(HOME_PATH, "musicfm", "data", "pretrained_msd.pt"),
        )
        stat_path=os.path.join(HOME_PATH, "musicfm", "data", "msd_stats.json"),
        model_path=os.path.join(HOME_PATH, "musicfm", "data", "pretrained_msd.pt"),
        print(stat_path)
        print(model_path)

        # # to GPUs
        # wav = wav.cuda()
        # musicfm = musicfm.cuda()

        # get embeddings
        musicfm.eval()
        emb = musicfm.get_latent(waveform, layer_ix=7)
        return emb
        