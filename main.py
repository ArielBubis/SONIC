import logging
import SONIC.CREAM as CREAM # Conversion, Resources, and Enhancements for Audio and Modules
import SONIC.TAILS as TAILS # Transformative Audio Intelligence for Learning and Spectrograms

import argparse

# Parse command line arguments for task and kwargs


if __name__ == "__main__":
    
    # Profiling
    profiler = CREAM.utils.start_profiler()

    parser = argparse.ArgumentParser(description='Simple Task Manager for Audio Processing')
    parser.add_argument('task', type=str, help='Task to perform: download or convert', choices=['convert', 'embed'])

    # for convert task
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, help='Directory to save generated spectrograms')
    # parser.add_argument('--max-workers', type=int, help='Number of concurrent workers')

    # for embedding task
    parser.add_argument('--model-type', type=str, help='Model type')
    parser.add_argument('--spectrograms-path', type=str, help='Path to the spectrograms')
    parser.add_argument('--stride', type=int, help='Stride length for extracting windows')

    args = parser.parse_args()
    task = args.task

    # audio_dir = 'music4all/music4all/test'
    # output_dir = 'output_test'

    # Convert pixel dimensions to inches
    if task == 'convert':
        audio_dir = args.audio_dir
        output_dir = args.output_dir
        dpi = 100
        fig_size = (448 / dpi, 224 / dpi)  # Adjust based on desired DPI

        # Execute main functionality
        CREAM.utils.setup_logging('audio_visualization.log')
        CREAM.convert.audio_to_spectograms(audio_dir, output_dir, fig_size, max_workers=8)

    elif task == 'embed':
        model_name = args.model_type
        spectrograms_path = args.spectrograms_path
        stride = args.stride
        spectrogram_type = spectrograms_path.split('/')[-1]

        # Execute main functionality
        CREAM.utils.setup_logging('audio_embedding.log')
        emb = TAILS.ViT.get_embeddings(model_name, spectrograms_path, stride)
        CREAM.utils.save_embeddings(emb, f'{model_name}_{spectrogram_type}_embeddings_{stride}_stride.csv')

    
    # Profiling
    CREAM.utils.stop_profiler(profiler, 'profile_data.prof')
