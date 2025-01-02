from functools import partial
import logging
import scipy as sp
import torch
import torch.nn as nn
import torchvision.transforms as T
import SONIC.CREAM as CREAM
import numpy as np
from tqdm.auto import tqdm
from SONIC.TAILS import embedder

VALID_ViT = ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16', 'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50']

WINDOW_SIZE = 224
STRIDE = 112 # 50% overlap (3 samples)
vit_preproc = T.Compose([
    T.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalization mean (ImageNet standard)
            std=[0.229, 0.224, 0.225]    # Normalization std (ImageNet standard)
        )
])

class ViTEmbedder(embedder.Embedder):
    def __init__(self, batch_size, model_name='dino_vits16', stride=STRIDE, level=0):
        super().__init__(batch_size)
        
        self.model = self.__get_vit_model(model_name).to(self.device)
        self.stride = stride
        self.level = level
        logging.info(f"Using model: {model_name}, stride: {stride}, embedding size: {self.model.num_features}, level: {level}")
    
    def embedding_fn(self, waveform):
        """
        Compute ViT embeddings for the given waveform.
        Parameters:
            waveform (torch.Tensor): Input waveform.
        Returns:
            torch.Tensor: ViT embedding.
        """
        spectrogram = CREAM.convert.waveform_to_image(waveform, sr=16_000, n_mels=224)
        spectrogram = spectrogram.to(self.device)
        windows = self.__extract_windows(spectrogram, self.stride)
        windows = vit_preproc(windows)
        return np.mean(self.__get_window_embeddings(self.model, windows, self.level), axis=0)
    
    def get_embeddings(self, audio_dir):
        dataloader = CREAM.dataset.init_dataset(audio_dir, transform=self.embedding_fn, batch_size=self.batch_size)
        embeddings = []

        for i,batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
            logging.info(f"Extracting ViT embeddings for batch {i+1}/{len(dataloader)}")
            for audio_path, embedding in zip(*batch):
                embeddings.append((audio_path, embedding))  # Append path and embedding
                logging.info(f"Extracted ViT embedding for {audio_path}")
        
        return embeddings
    
    def __get_vit_model(self, model_name: str):
        """
        Get the ViT model from the torch hub.
        Parameters:
            model_name (str): Model name.
        Returns:
            torch.nn.Module: ViT model.
        """
        assert model_name in VALID_ViT, f"Invalid model name. Choose from {VALID_ViT}"
        model = torch.hub.load('facebookresearch/dino:main', model_name)
        return model
    
    def __extract_windows(self, spectrogram: torch.Tensor, stride: int = STRIDE):
        """
        Extract sliding windows from the spectrogram.
        """
        _, _, width = spectrogram.shape
        windows = [
            spectrogram[:, :, i:i + WINDOW_SIZE]
            for i in range(0, width - WINDOW_SIZE + 1, stride)
        ]
        return torch.stack(windows)

    def __get_window_embeddings(self, model: nn.Module, windows: torch.Tensor, level: int):
        """
        Get embeddings for a batch of windows using the ViT model.
        """
        with torch.no_grad():
            if level == 0:
                embeddings = model(windows)
            else:
                features = model.get_intermediate_layers(windows, n=1)
                embeddings = features[0][:,0,:]
            # embeddings = model(windows)
        return embeddings.cpu().numpy()
