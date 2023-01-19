import os, gc, re, ast, sys, copy
import json, time, math, string, pickle
import random, joblib, itertools, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence 
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import polars as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        aid = [item[0] for item in batch]
        aid_type = [item[1] for item in batch]
        session_type_target = [item[2] for item in batch]
        session_aid_target = [item[3] for item in batch]
        
        aid = pad_sequence(aid, batch_first=True, padding_value=self.pad_idx)
        aid_type = pad_sequence(aid_type, batch_first=True, padding_value=self.pad_idx)
        session_type_target = pad_sequence(session_type_target, batch_first=True, padding_value=self.pad_idx)
        session_aid_target = pad_sequence(session_aid_target, batch_first=True, padding_value=self.pad_idx)

        return aid, aid_type, session_type_target, session_aid_target


class OttoTrainDataset(Dataset):

    def __init__(self, df_loc, session_ids, enc_window=3):
        df = pl.read_parquet(df_loc)
        self.session_ids = session_ids
        
        df_grouped =  df.filter(pl.col('session').is_in(self.session_ids)
                                ).sort(
                                    by='ts'
                                ).groupby(
                                    'session'
                                ).agg(
                                    [pl.col('aid').alias('aid_list'),
                                    pl.col('type').alias('type_list')]
                                )                    
        self.df_grouped = df_grouped.to_pandas()
        del df; gc.collect() 

        self.vocab = { "itos": {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"},
                       "stoi" : {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
                     } # "<UNK>": 3}
        self.spz_vocab_len = len(self.vocab['itos'].keys())   
        self.enc_window  = enc_window
    
    def __len__(self):
        return len(self.session_ids)
    
    def get_vocab_size(self):
        return 1855603 + 3

    def get_block_size(self):
        return 100
    
    def __getitem__(self, item):
        session_df = self.df_grouped.iloc[item]
        session_aid, session_type = session_df['aid_list'].tolist(), session_df['type_list'].tolist() 
        
        session_aid_enc = [self.vocab['stoi']['<SOS>']] + [i + self.spz_vocab_len for i in session_aid[0:self.enc_window]] + [self.vocab['stoi']['<EOS>']]  # 3 
        session_type_enc = [0] + [i+1 for i in session_type[0:self.enc_window]] + [0] # 0 is my a_type for sos, eos and pad , actuals a_type-> a_type_orig+1
        
        session_aid_dec = [self.vocab['stoi']['<SOS>']] + [i + self.spz_vocab_len for i in session_aid[self.enc_window:]] + [self.vocab['stoi']['<EOS>']]  # 3 
        session_type_dec = [0] + [i+1 for i in session_type[self.enc_window:]] + [0] # 0 is my a_type for sos, eos and pad , actuals a_type-> a_type_orig+1

        session_aid_enc = torch.tensor(session_aid_enc, dtype=torch.long)
        session_type_enc = torch.tensor(session_type_enc, dtype=torch.long)

        session_aid_dec = torch.tensor(session_aid_dec, dtype=torch.long)
        session_type_dec = torch.tensor(session_type_dec, dtype=torch.long)

        return session_aid_enc, session_type_enc, session_type_dec, session_aid_dec

def get_loader(df_loc,
    session_ids,
    batch_size=8,
    num_workers=8,
    shuffle=True,
    pin_memory=False,
    enc_window=3
):
    dataset = OttoTrainDataset(df_loc, session_ids, enc_window)
    pad_idx = dataset.vocab['stoi']["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    # may wanna use sorted by len batch and train
    loader, dataset = get_loader(df_loc='otto_custom/data/archive/train.parquet', session_ids=list(range(10000)))
    for idx, (session_aid, session_aid_type, session_type_target, session_aid_target) in enumerate(loader):
        print(session_aid.shape)
        print(session_aid_type.shape)