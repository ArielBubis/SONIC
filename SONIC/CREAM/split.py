import logging
import pandas as pd
from sklearn.model_selection import train_test_split

START_DATE = '2018-02-20'
TEST_DATE = '2019-01-20'
END_DATE = '2019-02-20'

def split_data(interactions_df_path: str, sep: str=',' ,start_date: str=START_DATE, test_date: str=TEST_DATE, end_date: str = END_DATE) -> None:
    """
    Split the interactions data into train, validation, and test sets, based on the given dates.
    Parameters:
        interactions_df_path (str): Path to the interactions data file.
        sep (str): Delimiter used in the interactions data file (csv only, default: ',').
        start_date (str): Start date for the train set.
        test_date (str): Start date for the test set.
    Returns:
        None
    """
    if interactions_df_path.endswith('.parquet') or interactions_df_path.endswith('.pqt'):
        df = pd.read_parquet(interactions_df_path)
    else:
        df = pd.read_csv(interactions_df_path, sep=sep)
        df.timestamp = pd.to_datetime(df.timestamp).dt.floor('min')

    df = df[(df.timestamp >= pd.to_datetime(start_date)) & (df.timestamp < pd.to_datetime(end_date))]

    te = df[df.timestamp >= pd.to_datetime(test_date)].reset_index(drop=True) # test set

    # This is done so that the number of plays in train doesn't account for plays that happened after `test_date`
    tr = df[df.timestamp < pd.to_datetime(test_date)].reset_index(drop=True) # train set

    te = te[te.user_id.isin(tr.user_id.unique())]
    te = te[te.track_id.isin(tr.track_id.unique())]


    validation_user_ids, test_user_ids = train_test_split(te.user_id.unique(), test_size=0.5, random_state=42)
    val = te[te.user_id.isin(validation_user_ids)].reset_index(drop=True)
    test = te[te.user_id.isin(test_user_ids)].reset_index(drop=True)

    logging.info(f"Interactions data loaded from {interactions_df_path}")
    logging.info(f"Data split based on the following dates:")
    logging.info(f"Start date: {start_date}")
    logging.info(f"End date: {end_date}")
    logging.info(f"Test date: {test_date}")
    logging.info(f"Train set size: {{{len(tr):,} interactions, {len(tr.user_id.unique()):,} users, {len(tr.track_id.unique()):,} tracks}}")
    logging.info(f"Validation set size: {{{len(val):,} interactions, {len(val.user_id.unique()):,} users, {len(val.track_id.unique()):,} tracks}}")
    logging.info(f"Test set size: {{{len(test):,} interactions, {len(test.user_id.unique()):,} users, {len(test.track_id.unique()):,} tracks}}")

    out_dir = interactions_df_path.rsplit('/', 1)[0]

    test.to_parquet(f'{out_dir}/test.pqt')
    val.to_parquet(f'{out_dir}/validation.pqt')
    tr.to_parquet(f'{out_dir}/train.pqt')