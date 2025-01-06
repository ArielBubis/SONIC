import pandas as pd
from sklearn.model_selection import train_test_split

START_DATE = '2019-02-20'
TEST_DATE = '2020-02-20'

def split_data(interactions_df_path: str, sep: str=',' ,start_date: str=START_DATE, test_date: str=TEST_DATE) -> None:
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

    compress = lambda x: x.groupby(['user_id', 'track_id']).agg(timestamp=("timestamp", "min"), count=("timestamp", "count")).reset_index()
    if interactions_df_path.endswith('.parquet') or interactions_df_path.endswith('.pqt'):
        df = pd.read_parquet(interactions_df_path)
    else:
        df = pd.read_csv(interactions_df_path, sep=sep)
        df.timestamp = pd.to_datetime(df.timestamp).dt.floor('min')

    df = df[df.timestamp >= pd.to_datetime(start_date)]
    a = compress(df)

    te = a[a.timestamp >= pd.to_datetime(test_date)]

    # This is done so that the number of plays in train doesn't account for plays that happened after `test_date`
    df = df[df.timestamp < pd.to_datetime(test_date)]
    tr = compress(df)

    te = te[te.user_id.isin(tr.user_id.unique())]
    te = te[te.track_id.isin(tr.track_id.unique())]


    validation_user_ids, test_user_ids = train_test_split(te.user_id.unique(), test_size=0.5, random_state=42)
    val = te[te.user_id.isin(validation_user_ids)]
    test = te[te.user_id.isin(test_user_ids)]

    test.to_parquet('test.pqt')
    val.to_parquet('val.pqt')
    tr.to_parquet('train.pqt')