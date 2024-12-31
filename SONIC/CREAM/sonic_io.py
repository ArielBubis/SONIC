import logging
import pandas as pd

def save_embeddings(embeddings, output_file):
    """
    Save the embeddings to a CSV file.

    Parameters:
        embeddings (torch.Tensor): Embeddings to save.
        output_file (str): Path to save the embeddings.
    """
    embeddings = pd.DataFrame(embeddings, columns=['spectrogram', 'embedding'])
    embeddings = pd.concat([embeddings.drop(columns=['embedding']), embeddings['embedding'].apply(pd.Series)], axis=1)
    embeddings.to_csv(output_file, index=False)
    logging.info(f"Embeddings saved to {output_file}")