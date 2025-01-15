import logging
import pandas as pd
import numpy as np
import os

def save_embeddings(embeddings, output_file, **kwargs):
    """
    Save the embeddings to a parquet file.

    Parameters:
        embeddings (torch.Tensor): Embeddings to save.
        output_file (str): Path to save the embeddings.
    """
    if not os.path.exists(output_file.rsplit('/', 1)[0]):
        os.makedirs(output_file.rsplit('/', 1)[0])
    embeddings = pd.DataFrame(embeddings, columns=['track_id', 'embedding'])
    embeddings = pd.concat([embeddings.drop(columns=['embedding']), embeddings['embedding'].apply(pd.Series)], axis=1).set_index('track_id')
    np_type = kwargs.get('np_type', np.float16)
    embeddings = embeddings.astype(np_type)
    embeddings.to_parquet(output_file)
    logging.info(f"Embeddings saved to {output_file}")