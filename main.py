import generateSpectogram

if __name__ == "__main__":
    audio_dir = 'music4all/music4all/test'  
    output_dir = 'audio_visualizations'
    
    # Optimize and process audio files
    generateSpectogram.optimize_audio_visualization(audio_dir, output_dir)