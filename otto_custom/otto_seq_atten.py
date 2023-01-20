import random
import torch
import torch.nn as nn
import torch.optim as optim
from custom_seq2_seq import *
# import numpy as np
from utils import save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
# from torchtext.datasets import Multi30k
# from torchtext.data import Field, BucketIterator

# """
# To install spacy languages do:
# python -m spacy download en
# python -m spacy download de
# """
# spacy_ger = spacy.load("de")
# spacy_eng = spacy.load("en")


# def tokenize_ger(text):
#     return [tok.text for tok in spacy_ger.tokenizer(text)]


# def tokenize_eng(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]


# german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

# english = Field(
#     tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
# )

# train_data, valid_data, test_data = Multi30k.splits(
#     exts=(".de", ".en"), fields=(german, english)
# )

# german.build_vocab(train_data, max_size=10000, min_freq=2)
# english.build_vocab(train_data, max_size=10000, min_freq=2)


### We're ready to define everything we need for training our Seq2Seq model ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 10
learning_rate = 3e-4
batch_size = 16

# Model hyperparameters
a_list_size = 1855606
# encoder_embedding_size = 512
# decoder_embedding_size = 512
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0


embedding_size = 64
embedding_shared = nn.Embedding(20000, embedding_size) # 1855606
a_type_emb_shared= nn.Embedding(4, embedding_size)

encoder_net = Encoder(
    embedding_size, embedding_shared, a_type_emb_shared, hidden_size, num_layers, enc_dropout
)
#.to(device)

decoder_net = Decoder(
    embedding_size, 
    embedding_shared, 
    a_type_emb_shared,
    hidden_size,
    a_list_size,
    num_layers,
    dec_dropout,
)
#.to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

from otto_dataset_simple import get_loader
from otto_datasplit import OttoDataSplit
import pandas as pd
import gc

train_df = pd.read_parquet('otto_custom/data/archive/train.parquet')
otto_split = OttoDataSplit(train_df, min_session_len=3, max_session_len=100)
# use any type of split to get session distribution for training
train_session_ids, test_session_ids = otto_split.get_split_session_ids(train_size=0.7)
    
loader, dataset = get_loader(df_loc='otto_custom/data/archive/train.parquet', session_ids=train_session_ids, enc_window=3)
del train_df; del otto_split; del train_session_ids; del test_session_ids
gc.collect()

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
    model.train()

    for batch_idx, session_aid, session_type, session_type_target, session_aid_target in enumerate(loader):
        # Get input and targets and get to cuda
        
        session_aid = session_aid # .to(device)
        session_type = session_type.to(device)
        session_type_target = session_type_target.to(device)
        session_aid_target = session_aid_target.to(device)
        
        # Forward prop
        output, outputs_a_type = model(session_aid, session_type, session_type_target, session_aid_target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        session_aid_target = session_aid_target[1:].reshape(-1)

        outputs_a_type = outputs_a_type[1:].reshape(-1, outputs_a_type.shape[2])
        session_type_target = session_type_target[1:].reshape(-1)

        optimizer.zero_grad()
        loss_aid = criterion(output, session_aid_target)
        loss_atype = criterion(outputs_a_type, session_type_target)

        loss = loss_aid + loss_atype
        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

# running on entire test data takes a while
# score = bleu(test_data[1:100], model, german, english, device)
# print(f"Bleu score {score * 100:.2f}")