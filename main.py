import logging
import SONIC.CREAM as CREAM # Conversion, Resources, and Enhancements for Audio and Modules
import SONIC.TAILS as TAILS # Transformative Audio Intelligence for Learning and Spectrograms

import argparse

# Parse command line arguments for task and kwargs


if __name__ == "__main__":
    
    # Profiling
    profiler = CREAM.utils.start_profiler()

    parser = argparse.ArgumentParser(description='Simple Task Manager for Audio Processing')
    parser.add_argument('task', type=str, help='Task to perform: download or convert', choices=['convert', 'embedding'])

    # for convert task
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output-dir', type=str, help='Directory to save generated spectrograms')
    # parser.add_argument('--max-workers', type=int, help='Number of concurrent workers')

    # for embedding task
    parser.add_argument('--model-type', type=str, help='Model type')
    parser.add_argument('--spectrogram-path', type=str, help='Path to the spectrogram')
    parser.add_argument('--hop-length', type=int, help='Hop length for the audio file')

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

    elif task == 'embedding':
        model_name = args.model_type
        spectoram_path = args.spectrogram_path
        hop_length = args.hop_length

        # Execute main functionality
        CREAM.utils.setup_logging('audio_embedding.log')
        emb = TAILS.ViT.get_vit_embedding('dino_vits16', spectoram_path, 224)
        logging.info(f"Embedding shape: {emb.shape}, values: {emb}")

    
    # Profiling
    CREAM.utils.stop_profiler(profiler, 'profile_data.prof')
