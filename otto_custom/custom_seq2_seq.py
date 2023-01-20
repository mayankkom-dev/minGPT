import random
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

