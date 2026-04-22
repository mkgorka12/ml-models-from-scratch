import pandas as pd


def fill_dataframe_random_bools(rng, df, number_records):
    data = rng.binomial(n=1, p=0.5, size=(number_records, df.shape[1])).astype(bool)
    return pd.DataFrame(data, columns=df.columns)
