import generateSpectogram

if __name__ == "__main__":
    # audio_dir = 'music4all/music4all/audios'  
    # output_dir = 'audio_visualizations'
    audio_dir = 'music4all/music4all/test'  
    output_dir = 'output_test'

    # Convert pixel dimensions to inches
    dpi = 100
    fig_size = (488 / dpi, 244 / dpi)  # Adjust based on desired DPI
    
    generateSpectogram.audio_visualization(audio_dir, output_dir, fig_size, max_workers=8)