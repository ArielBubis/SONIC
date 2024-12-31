from . import CREAM
from . import TAILS

import os
import argparse

def main():

    parser = argparse.ArgumentParser(description='Simple Task Manager for Audio Processing')
    parser.add_argument('task', type=str, help='Tasks to perform: embed', choices=['embed']) # TODO: Add more tasks
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--profile', action='store_true', help='Enable profiling', default=False)

    # kwargs for embedding task
    parser.add_argument('--model-type', type=str, help='Model type')
    parser.add_argument('--stride', type=int, help='Stride length for extracting windows')

    args = parser.parse_args()
    task = args.task
    profiling = args.profile
    if profiling:
        profiler = CREAM.utils.start_profiler()

    # TODO: remove by the next iteration
    # if task == 'convert':
    #     audio_dir = args.audio_dir
    #     output_dir = args.output_dir
    #     dpi = 100
    #     fig_size = (448 / dpi, 224 / dpi)  # Adjust based on desired DPI

    #     # Execute main functionality
    #     CREAM.utils.setup_logging('audio_visualization.log')
    #     CREAM.convert.audio_to_spectograms(audio_dir, output_dir, fig_size, max_workers=8)

    if task == 'embed':
        model_name = args.model_type
        audio_path = args.audio_dir
        if not os.path.exists(audio_path):
            print(f"Audio directory {audio_path} does not exist")
            return
        stride = args.stride

        # Execute main functionality
        CREAM.utils.setup_logging('audio_embedding.log')
        if 'mfcc' in model_name.lower():
            emb = TAILS.MFCC.get_embeddings(audio_path)
        
        # TODO: Implement the following
        # elif 'vit' in model_name.lower():
        #     emb = TAILS.ViT.get_embeddings(model_name, audio_path, stride)
        # elif 'musicnn' in model_name.lower():
        #     emb = TAILS.MusiCNN.get_embeddings(audio_path)

        CREAM.io.save_embeddings(emb, f'{model_name}_embeddings.csv')

    if profiling:
        CREAM.utils.stop_profiler(profiler, 'profile_data.prof')


if __name__ == '__main__':
    main()