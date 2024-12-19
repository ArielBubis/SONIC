import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_image_with_cv2(fig, output_filepath):
    """
    Save a matplotlib figure as an image using OpenCV.

    Parameters:
        fig (matplotlib.figure.Figure): Matplotlib figure to save.
        output_filepath (str): Path to save the image.
    """
    fig.canvas.draw()
    img_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filepath, img_plot)
    plt.close(fig)

def calculate_dynamic_hop_length(n_fft, sample_rate, target_time_resolution=0.001):
    """
    Calculate a hop length based on the target time resolution.
    Ensures adequate overlap between FFT windows for smooth spectrograms.

    Parameters:
        n_fft (int): Number of FFT components.
        sample_rate (int): Audio sample rate in Hz.
        target_time_resolution (float): Desired time resolution in seconds.

    Returns:
        int: Computed hop length.
    """
    return max(1, int(target_time_resolution * sample_rate))
