import cProfile
import logging

def setup_logging(log_file):
    """
    Setup logging for the application.

    Parameters:
        log_file (str): Path to the log file.
    """
    # Clear log file
    open(log_file, 'w').close()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )

def start_profiler():
    """
    Set up the profiler for the application.
    """
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    return profiler

def stop_profiler(profiler, output_file):
    """
    Stop the profiler and save the profiling data to a file.

    Parameters:
        profiler (cProfile.Profile): Profiler object.
        output_file (str): Path to save the profiling data.
    """
    profiler.disable()  # Stop profiling
    profiler.dump_stats(output_file)
    logging.info(f"Profiling data saved to {output_file}")
    # Optional: Launch SnakeViz visualization directly
    print("Run the following command to visualize the profiling data with SnakeViz:")
    print("snakeviz profile_data.prof")