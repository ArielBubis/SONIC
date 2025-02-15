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

def save_exclude_list(exclude_list, output_file = 'data/exclude.pqt'):
    """
    Save the exclude list to a file.

    Parameters:
        exclude_list (list): List of track IDs to exclude.
        output_file (str): Path to save the exclude list.
    """
    exclude = pd.DataFrame(exclude_list, columns=['track_id'])
    exclude['track_id'] = exclude['track_id'].str.split('/').str[-1].str.split('.').str[0]
    exclude.to_parquet(output_file, index=False)
    logging.info(f"Exclude list saved to {output_file}")