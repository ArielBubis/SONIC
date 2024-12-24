import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import CREAM
from PIL import Image
import numpy as np

VALID_ViT = ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16', 'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50']

WINDOW_SIZE = 224
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

def __load_image(image_path: str):
    """
    Load an image from the given path.
    Parameters:
        image_path (str): Path to the image.
    Returns:
        torch.Tensor: Image tensor.
    """
    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image)
    logging.info(f"Loaded image from {image_path}, shape: {image.shape}")
    return image


def __get_window_embedding(model: nn.Module, window: torch.Tensor):
    """
    Get the embedding of a window using a Vision Transformer.
    Parameters:
        model (torch.nn.Module): Vision Transformer model.
        window (torch.Tensor): Window to embed.
    Returns:
        torch.Tensor: Window embedding.
    """
    # ViT models require input shape (224, 224, 3, x) where x is the number of windows
    # assert tuple(window.shape[:3]) == (224, 224, 3), f"Input shape must be (224, 224, 3, x), received {tuple(window.shape)}"
    with torch.no_grad():
        embedding = model(window)
    return embedding.cpu().numpy()

def get_vit_embedding(model: nn.Module, spectogram: torch.Tensor, hop_length: int):
    """
    Get the embedding of a window using a Vision Transformer.
    Parameters:
        model (torch.nn.Module): Vision Transformer model.
        spectogram (torch.Tensor): Spectogram to embed.
        hop_length (int): Hop length for the audio file.
    Returns:
        torch.Tensor: Window embedding.
    """
    window_embeddings = []
    for i in range(0, len(spectogram), hop_length):
        # take a 224x224 window
        window = spectogram[:, :, i:i + WINDOW_SIZE]
        window = vit_preproc(window).unsqueeze(0)
        window_embedding = __get_window_embedding(model, window)
        window_embeddings.append(window_embedding)
    return np.mean(window_embeddings, axis=0)


def get_embeddings(model_name: str, spectograms_dir: str, hop_length: int):
    """
    Get the embeddings of all windows in a directory of spectrograms.
    Parameters:
        model_name (str): Model name.
        spectograms_dir (str): Path to the directory containing spectrograms.
        hop_length (int): Hop length for the audio file.
    Returns:
        np.ndarray: Embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = __get_vit_model(model_name).to(device)
    dataset = CREAM.init_dataset(spectograms_dir)
    embeddings = []
    for spectogram in dataset:
        spectogram = spectogram.to(device)
        embeddings = get_vit_embedding(model, spectogram, hop_length)
    return np.array(embeddings)