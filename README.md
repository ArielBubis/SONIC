# SONIC: (S)pectrogram-(O)riented (N)etwork for (I)ntelligent Re(C)ommendation
## Overview
A comprehensive audio processing pipeline that transforms music files into visual representations for machine learning-based music recommendation. Part of a Recommender Systems course project, this tool generates multi-dimensional audio visualizations to enable image-based music categorization and analysis.
python version should be Python 3.12.2
## Dataset
*Dataset Credit*: [Music4All Dataset](https://ieeexplore.ieee.org/document/9145170)

Special thanks to the original dataset creators for providing the music collection used in this research.

## Module Architecture
The SONIC pipeline consists of the following modules:
- `CREAM` (Conversion, Resources and Enrichment for Audio and Modules): This is a utility module that provides the necessary resources and functions for the SONIC pipeline.
- `TAILS` (Transformations and Analysis for Intelligent Learning Systems): This module is responsible for embedding the audio spectrograms.
- `SHADOW` (Systematic Hyperparameter Analysis and Dynamic Optimization Workflow): This module is responsible for the hyperparameter tuning and optimization of the SONIC pipeline.

<!--- more modules to be added here --->
## Key Features

Multi-threaded audio file processing
Four advanced audio visualization techniques:
* Mel Spectrograms
* Chromagrams
* FFT Spectra
* Tempograms


Optimized performance with parallel processing
Flexible configuration for audio analysis

## Visualizations
The pipeline creates rich, informative visual representations of audio files:

* Mel Spectrograms: Frequency-based energy distribution
![00b6fV3nx5z2b8Ls](https://github.com/user-attachments/assets/0355705e-dc46-4b3c-81a0-75282e1d9fea)
![00CH4HJdxQQQbJfu](https://github.com/user-attachments/assets/f2580091-f44d-4f86-b454-86a018dd58a3)

* Chromagrams: Harmonic and tonal content
![00b6fV3nx5z2b8Ls](https://github.com/user-attachments/assets/81882d00-cb0f-4c20-b252-564d5c20a62a)
![00CH4HJdxQQQbJfu](https://github.com/user-attachments/assets/3bcd77b4-bcca-4b8d-a81a-313e02c47c8b)

* FFT Spectra: Frequency amplitude analysis
![00b6fV3nx5z2b8Ls](https://github.com/user-attachments/assets/9e111706-ea44-41b8-9687-ddc9f5ad25d7)
![00CH4HJdxQQQbJfu](https://github.com/user-attachments/assets/c7cf8400-b9b2-4392-90e5-b0f1e61598df)

* Tempograms: Rhythmic pattern visualization
![00b6fV3nx5z2b8Ls_tempogram](https://github.com/user-attachments/assets/b3580acc-f11d-4f6d-9490-a63a9cf9600a)
![00CH4HJdxQQQbJfu_tempogram](https://github.com/user-attachments/assets/4c06722b-ac88-451c-8598-6f5028f3200b)

## Purpose
Transform audio data into machine learning-ready image datasets for:

* Music genre classification
* Recommendation system training
* Advanced audio feature extraction

## Technologies

* Python 
* librosa
* NumPy
* Matplotlib
* OpenCV
* Pytorch
* Multiprocessing

## Getting Started

1. Install the package using pip:
```bash
pip install git+https://git@github.com:ArielBubis/sound_visualisation_for_music_recSys.git
```
2. Import the package:
```python
import sonic
```
3. use cli to run the code
```bash
sonic --help
```

Developed as part of a Recommender Systems course project.
