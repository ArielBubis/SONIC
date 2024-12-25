import logging
from math import e
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import CREAM
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

VALID_ViT = ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16', 'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50']

WINDOW_SIZE = 224
STRIDE = 112 # 50% overlap (3 samples)
vit_preproc = transforms.Compose([
    transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalization mean (ImageNet standard)
            std=[0.229, 0.224, 0.225]    # Normalization std (ImageNet standard)
        )
])

def __get_vit_model(model_name: str):
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

def __extract_windows(spectrogram: torch.Tensor, stride: int = STRIDE):
    """
    Extract sliding windows from the spectrogram.
    """
    _, _, width = spectrogram.shape
    windows = [
        spectrogram[:, :, i:i + WINDOW_SIZE]
        for i in range(0, width - WINDOW_SIZE + 1, stride)
    ]
    return torch.stack(windows)

def __get_window_embeddings(model: nn.Module, windows: torch.Tensor, level: int):
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

def __get_vit_embedding(model: nn.Module, spectrogram: torch.Tensor, stride: int, level: int):
    """
    Get embeddings for all windows in a spectrogram.
    """
    windows = __extract_windows(spectrogram, stride)
    windows = vit_preproc(windows)
    return np.mean(__get_window_embeddings(model, windows, level), axis=0)

def get_embeddings(model_name: str, spectograms_dir: str, stride: int = STRIDE, level: int = 0):
    """
    Get embeddings for all spectrograms in a directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = __get_vit_model(model_name).to(device)
    logging.info(f"Using device: {device}")
    logging.info(f"""
                 Using model: {model_name}, 
                 stride: {stride}, 
                 embedding size: {model.num_features}, 
                 level: {level}""")
    dataset = CREAM.utils.init_dataset(spectograms_dir)
    embeddings = []

    for i,batch in tqdm(enumerate(dataset), desc="Extracting embeddings", total=len(dataset)):
        logging.info(f"Extracting ViT embeddings for batch {i+1}/{len(dataset)}")
        for spectrogram_path, spectrogram in zip(*batch):
            logging.info(f"Extracting ViT embedding for {spectrogram_path}")
            spectrogram = spectrogram.to(device)  # Move spectrogram to device
            embedding = __get_vit_embedding(model, spectrogram, stride, level)  # Get embedding
            embeddings.append((spectrogram_path, embedding))  # Append path and embedding
        
    
    return embeddings