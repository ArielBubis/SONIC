from email.mime import audio
import CAPAR_MRSim.file_processing.convert as convert
import CAPAR_MRSim.file_processing.download as download
import cProfile

import argparse

# Parse command line arguments for task and kwargs


if __name__ == "__main__":
    # Profiling setup
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    parser = argparse.ArgumentParser(description='Simple Task Manager for Audio Processing')
    parser.add_argument('task', type=str, help='Task to perform: download or convert', choices=['download', 'convert'])

    # for convert task
    parser.add_argument('--audio_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, help='Directory to save generated spectrograms')
    parser.add_argument('--fig_size', type=str, help='Size of the figure')
    parser.add_argument('--max_workers', type=int, help='Number of concurrent workers')

    # for download task
    parser.add_argument('--file', type=str, help='Path to torrent file')
    parser.add_argument('--output', type=str, help='Output directory for downloaded files')

    args = parser.parse_args()
    task = args.task
    if task == 'download':
        file = args.file
        output = args.output
        download.download_torrent(file, output)

    # audio_dir = 'music4all/music4all/test'
    # output_dir = 'output_test'

    # Convert pixel dimensions to inches
    else:
        audio_dir = args.audio_dir
        output_dir = args.output_dir
        dpi = 100
        fig_size = (488 / dpi, 244 / dpi)  # Adjust based on desired DPI

        # Execute main functionality
        convert.audio_to_spectograms(audio_dir, output_dir, fig_size, max_workers=8)

    profiler.disable()  # Stop profiling

    # Save profiling data to file
    profiler.disable()
    profiler.dump_stats("profile_data.prof")

    # Optional: Launch SnakeViz visualization directly
    print("Run the following command to visualize the profiling data with SnakeViz:")
    print("snakeviz profile_data.prof")
