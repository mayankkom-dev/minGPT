import random
import torch
import torch.nn as nn
import torch.optim as optim
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

embedding_size = 1024
embedding_shared = nn.Embedding(1855606, embedding_size)
a_type_emb_shared= nn.Embedding(4, embedding_size)

class Encoder(nn.Module):
    def __init__(self, embedding_size, embedding_shared, a_type_emb_shared, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding_shared
        self.a_type_emb = a_type_emb_shared 
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, a_list_src, a_type_list_src):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(a_list_src))
        # embedding shape: (seq_length, N, embedding_size)

        a_type_embedding = self.dropout(self.a_type_emb(a_type_list_src))
        # add to embedding size
        combined_emb = self.dropout(embedding + a_type_embedding)

        encoder_states, (hidden, cell) = self.rnn(combined_emb)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, embedding_size, embedding_shared, a_type_emb_shared, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding_shared
        self.a_type_emb = a_type_emb_shared

        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.a_type_fc = nn.Linear(hidden_size, 4)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, a_list_0, a_type_list_0, encoder_states, hidden, cell):
        a_list_0 = a_list_0.unsqueeze(0)
        # x: (1, N) where N is the batch size
        a_type_list_0 = a_type_list_0.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(a_list_0))
        # embedding shape: (seq_length, N, embedding_size)

        a_type_embedding = self.dropout(self.a_type_emb(a_type_list_0))
        # add to embedding size
        combined_emb = self.dropout(embedding + a_type_embedding)

        
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, combined_emb), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)
        pred_a_type = self.a_type_fc(outputs).squeeze(0)

        return predictions, pred_a_type, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, a_list_src, a_type_list_src, a_type_list_tgt, a_list_tgt, teacher_force_ratio=0.5):
        batch_size = a_list_src.shape[1]
        target_len = a_list_tgt.shape[0]
        a_list_vocab_size = 1855606

        outputs = torch.zeros(target_len, batch_size, a_list_vocab_size).to(device)
        outputs_a_type = torch.zeros(target_len, batch_size, 4).to(device)

        encoder_states, hidden, cell = self.encoder(a_list_src, a_type_list_src)

        # First input will be <SOS> token
        a_list_0 = a_list_tgt[0]
        a_type_0 = a_type_list_tgt[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, out_a_type, hidden, cell = self.decoder(a_list_0, a_type_0, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output
            outputs_a_type[t] = out_a_type
            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            best_guess_a_type = outputs_a_type.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            a_list_0 = a_list_tgt[t] if random.random() < teacher_force_ratio else best_guess
            a_type_0 = a_type_list_tgt[t] if random.random() < teacher_force_ratio else best_guess_a_type
        return outputs, outputs_a_type


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


encoder_net = Encoder(
    embedding_size, embedding_shared, a_type_emb_shared, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    embedding_size, 
    embedding_shared, 
    a_type_emb_shared,
    hidden_size,
    a_list_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = 0
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

from otto_dataset import get_loader
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
        
        session_aid = session_aid.to(device)
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