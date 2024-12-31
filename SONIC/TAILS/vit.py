from functools import partial
import logging
import scipy as sp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import SONIC.CREAM as CREAM
import numpy as np
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

def vit_embedding(waveform: torch.Tensor, model: nn.Module, stride: int, level: int, device: torch.device):
    """
    Get embeddings for all windows in a spectrogram.
    """
    spectrogram = CREAM.convert.waveform_to_image(waveform, sr=16_000, n_mels=224)
    spectrogram = spectrogram.to(device)
    windows = __extract_windows(spectrogram, stride)
    windows = vit_preproc(windows)
    return np.mean(__get_window_embeddings(model, windows, level), axis=0)


def get_embeddings(model_name: str, audio_dir: str, stride: int = STRIDE, level: int = 0):
    """
    Get embeddings for all spectrograms in a directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit = __get_vit_model(model_name).to(device)
    vit_fn = partial(vit_embedding, model=vit, stride=stride, level=level, device=device)
    logging.info(f"Using device: {device}")
    logging.info(f"""
                 Using model: {model_name}, 
                 stride: {stride}, 
                 embedding size: {vit.num_features}, 
                 level: {level}""")
    dataloader = CREAM.dataset.init_dataset(audio_dir, transform=vit_fn)
    embeddings = []

    for i,batch in tqdm(enumerate(dataloader), desc="Extracting embeddings", total=len(dataloader)):
        logging.info(f"Extracting ViT embeddings for batch {i+1}/{len(dataloader)}")
        for audio_path, embedding in zip(*batch):
            embeddings.append((audio_path, embedding))  # Append path and embedding
            logging.info(f"Extracted ViT embedding for {audio_path}")
    
    return embeddings