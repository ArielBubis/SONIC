import matplotlib.pyplot as plt
from PIL import Image

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