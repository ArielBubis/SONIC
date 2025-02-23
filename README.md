# SONIC: Cross-Modal Music Recommendation System

## Overview

SONIC (Spectrogram-Oriented Network for Intelligent reCommendation) is a research project investigating the effectiveness of pre-trained audio representations in music recommendation systems (MRS). The project uniquely explores not only traditional audio embedders but also applies vision transformers (ViT) to music recommendation through spectrogram analysis and integrates multilingual text embeddings for lyrics processing.

### Research Motivation
Traditional music recommendation systems face challenges like the cold-start problem and popularity bias. While hybrid approaches combining collaborative and content-based filtering exist, they often rely heavily on manually curated metadata. SONIC addresses these limitations by leveraging pre-trained models to automatically capture key musical elements like timbre, rhythm, and melody without manual feature engineering.

### Key Research Questions
1. How do different backend models compare in their recommendation performance?
2. How do general-purpose embedders like DINO-ViT compare to audio-based embedders?
3. How does concatenating lyrical embeddings impact recommendation performance?
4. How does integrating additional track metadata affect MRS metrics?
5. How does model performance vary at different k-values?

## Key Features

- **Cross-Modal Analysis**: Combines audio, visual, and textual representations for comprehensive music understanding
- **Multiple Embedding Approaches**:
  - Audio-specific models (MFCC, MERT, EncodecMAE)
  - Vision Transformers (DINO-ViT) applied to spectrograms
  - Multilingual text embeddings for lyrics
- **Advanced Audio Visualizations**:
  - Mel Spectrograms for frequency analysis
  - Chromagrams for harmonic content
  - FFT Spectra for frequency distribution
  - Tempograms for rhythm patterns
- **Efficient Processing**:
  - Multi-threaded audio processing
  - Optimized parallel computation
  - Flexible configuration options

## Architecture

SONIC consists of three main modules:

### 1. CREAM (Conversion, Resources and Enrichment for Audio and Modules)
- Audio file handling and conversion
- Resource management
- Utility functions and enrichment processes

### 2. TAILS (Transformations and Analysis for Intelligent Learning Systems)
- Multiple embedding pipelines:
  - MFCC (Mel-frequency cepstral coefficients)
  - MERT (Music Encoder Representations from Transformers)
  - EncodecMAE (Audio Codec with Masked Autoencoder)
  - DINO-ViT (Vision Transformer for spectrogram analysis)
  - Lyrical embeddings (multilingual text processing)

### 3. ROUGE (Recommendation Optimization and User Guidance Engine)
- Recommendation algorithms:
  - KNN (K-Nearest Neighbors)
  - SNN (Shallow Neural Networks)
  - BERT4Rec (BERT for sequential recommendation)

## Installation

1. Ensure you have Python 3.12.2 installed
2. Install the package:
```bash
pip install git+https://github.com/ArielBubis/sound_visualisation_for_music_recSys.git
```

## Usage

### Basic Usage
```python
import sonic

# Initialize the SONIC pipeline
sonic.CREAM  # For audio processing and utilities
sonic.TAILS  # For embedding generation
sonic.ROUGE  # For recommendation generation
```

### Command Line Interface
```bash
# Get help
sonic --help

# Generate embeddings
sonic embed --audio-dir /path/to/audio --model-type MFCC --batch-size 32

# Run recommendation model
sonic run-model --model-name knn --embedding mfcc_104 --mode val --k 50
```

## Model Types

### Audio Embedders
- **MFCC**: Traditional audio feature extraction
- **MERT**: Transformer-based audio understanding
- **EncodecMAE**: Advanced audio codec with masked autoencoding
- **M2V**: Music2Vec embeddings

### Vision Models
- **DINO-ViT**: Vision Transformer variants
  - `dino_vits16`
  - `dino_vits8`
  - `dino_vitb16`
  - `dino_vitb8`

### Text Processing
- **Lyrical**: Multilingual text embeddings for song lyrics
  - `paraphrase-multilingual-MiniLM-L12-v2`

## Requirements
- Python ≥3.10, <3.13
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- FAISS
- Transformers
- Additional dependencies listed in pyproject.toml

## Documentation

### Data Split Configuration
```python
# Default date ranges for data splitting
START_DATE = '2018-02-20'
TEST_DATE = '2019-01-20'
END_DATE = '2019-02-20'
```

### Artifact Generation
The system generates various visualizations for each audio file:
- Mel Spectrograms: Frequency-based energy distribution
<!-- - Chromagrams: Harmonic and tonal content analysis
- FFT Spectra: Detailed frequency analysis
- Tempograms: Rhythm and tempo visualization -->

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Citation

If you use this work in your research, please cite:
```
[Citation information to be added]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work builds upon and extends:

- "Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems" by Yan-Martin Tamm and Anna Aljanaki (RecSys '24) [arXiv:2409.08987](https://arxiv.org/abs/2409.08987)

- Based on the Music4All Dataset: [IEEE Paper](https://ieeexplore.ieee.org/document/9145170)
- Based on 
- Special thanks to the original dataset creators
