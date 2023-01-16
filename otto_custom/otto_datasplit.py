from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class OttoDataSplit:
    
    def __init__(self, df,  min_session_len, max_session_len):
        self.df = df
        self.min_session_len = min_session_len
        self.max_session_len = max_session_len

    def get_split_session_ids(self, train_size=0.7, random_state=29):
        df_grouped = self.df.groupby('session')
        df_grouped_size = df_grouped.size()
        all_session_ids = df_grouped_size[(df_grouped_size >= self.min_session_len) & (df_grouped_size <= self.max_session_len) ].reset_index()['session'].tolist() 
        #.sort_values(by=0, ascending=False)
        train_session_ids, test_session_ids = train_test_split(all_session_ids, train_size=train_size, random_state=random_state)
        return train_session_ids, test_session_ids

if __name__ == "__main__":
    train_df = pd.read_parquet('archive/train.parquet')
    otto_split = OttoDataSplit(train_df, min_session_len=3, max_session_len=100)
    # use any type of split to get session distribution for training
    train_session_ids, test_session_ids = otto_split.get_split_session_ids(train_size=0.7)
    print('Done')
