import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

# def save_image_with_cv2(fig, output_filepath):
#     """
#     saving using direct buffer access and minimal conversions.
#     """
#     fig.canvas.draw()
    
#     # Get the raw buffer and reshape it efficiently
#     buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#     img_plot = buffer.reshape((fig.canvas.get_width_height()[1], 
#                              fig.canvas.get_width_height()[0], 
#                              4))[:, :, :3]
    
#     # Direct BGR conversion without intermediate RGB step
#     img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
    
#     # Use optimal compression parameters
#     params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # Adjust quality as needed
#     cv2.imwrite(output_filepath, img_plot, params)
#     plt.close(fig)

def save_image_with_pillow(fig, output_filepath):
    """
    Alternative saving method using PIL/Pillow for potentially faster saving.
    """
    fig.canvas.draw()
    
    # Get the RGBA buffer
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    
    # Convert to PIL Image directly from buffer
    image = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)
    
    # Convert RGBA to RGB and save
    image.convert('RGB').save(output_filepath, 'JPEG', quality=95, optimize=True)
    plt.close(fig)

def calculate_dynamic_hop_length(sample_rate, target_time_resolution=0.001):
    """
    Calculate a hop length based on the target time resolution.
    Ensures adequate overlap between FFT windows for smooth spectrograms.

    Parameters:
        sample_rate (int): Audio sample rate in Hz.
        target_time_resolution (float): Desired time resolution in seconds.

    Returns:
        int: Computed hop length.
    """
    return max(1, int(target_time_resolution * sample_rate))