import generateSpectogram
import cProfile

if __name__ == "__main__":
    # Profiling setup
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    audio_dir = 'music4all/music4all/audios'
    output_dir = 'audio_visualization_data'
    # audio_dir = 'music4all/music4all/test'
    # output_dir = 'output_test'

    # Convert pixel dimensions to inches
    dpi = 100
    fig_size = (488 / dpi, 244 / dpi)  # Adjust based on desired DPI

    # Execute main functionality
    generateSpectogram.audio_visualization(audio_dir, output_dir, fig_size, max_workers=8)

    profiler.disable()  # Stop profiling

    # Save profiling data to file
    profiler.disable()
    profiler.dump_stats("profile_data.prof")

    # Optional: Launch SnakeViz visualization directly
    print("Run the following command to visualize the profiling data with SnakeViz:")
    print("snakeviz profile_data.prof")
