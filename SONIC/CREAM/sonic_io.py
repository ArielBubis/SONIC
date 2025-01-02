import logging
import pandas as pd

def save_embeddings(embeddings, output_file):
    """
    Save the embeddings to a parquet file.

    Parameters:
        embeddings (torch.Tensor): Embeddings to save.
        output_file (str): Path to save the embeddings.
    """
    embeddings = pd.DataFrame(embeddings, columns=['track_id', 'embedding'])
    embeddings = pd.concat([embeddings.drop(columns=['embedding']), embeddings['embedding'].apply(pd.Series)], axis=1).set_index('track_id')
    embeddings = embeddings.astype('float16')
    embeddings.to_parquet(output_file, index=False)
    logging.info(f"Embeddings saved to {output_file}")