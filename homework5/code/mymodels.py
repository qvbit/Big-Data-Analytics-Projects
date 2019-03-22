import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F




class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.hidden1 = nn.Linear(178, 64)
        self.hidden2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv1_bn = nn.BatchNorm1d(6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv2_bn = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(in_features=16*41, out_features=128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 5)
        
    def forward(self, x):
#         x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
#         x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
#         x = x.view(-1, 16*41) # Flatten the output to a vector feed into dense layer
#         x = F.relu(self.fc1_bn(self.fc1(x)))
#         x = self.fc2(x)

        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 16*41) # Flatten the output to a vector feed into dense layer
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout=0.7, bidirectional=True) #(batch, seq, features)
        self.fc = nn.Linear(in_features=32, out_features=5)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :]) # We only pass in the final time step hidden state to the dense layer.
        return x


# class MyVariableRNN(nn.Module):
#     def __init__(self, dim_input):
#         super(MyVariableRNN, self).__init__()
#         # You may use the input argument 'dim_input', which is basically the number of features
#         self.fc_size = 32
#         self.rnn_size = 16
#         self.num_layers = 2
#         self.bidirectional = False
#         self.dropout = 0.5

#         self.fc1 = nn.Linear(in_features=dim_input, out_features=self.fc_size)
#         torch.nn.init.xavier_uniform(self.fc1.weight)
#         self.drop = nn.Dropout(p = 0.5)

#         self.gru = nn.GRU(input_size=self.fc_size, hidden_size=self.rnn_size, num_layers=self.num_layers, batch_first=True,
#             bidirectional=self.bidirectional, dropout=self.dropout) # change input size
    
        
#         # if self.bidirectional:
#         #     self.rnn_size = self.rnn_size * 2

#         self.fc2 = nn.Linear(in_features=self.rnn_size, out_features=2)
#         torch.nn.init.xavier_uniform(self.fc2.weight)

#     def forward(self, input_tuple):
#         # HINT: Following two methods might be useful
#         # 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
#         seqs, lengths = input_tuple
        
#         batch_size = len(seqs)
#         seqs = F.relu(self.drop(self.fc1(seqs)))

#         seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
#         _, seqs = self.gru(seqs)
#         seqs = seqs[-1]

#         seqs = seqs.view(batch_size, -1)
#         seqs = self.fc2(seqs)
    
#         return seqs


class MyVariableRNN(nn.Module):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        # You may use the input argument 'dim_input', which is basically the number of features
        self.fc_size = 64
        self.rnn_size = 16
        self.num_layers = 2
        self.bidirectional = False
        self.dropout = 0.5

        self.fc1 = nn.Linear(in_features=dim_input, out_features=self.fc_size)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.drop = nn.Dropout(p = 0.5)

        self.gru = nn.GRU(input_size=self.fc_size, hidden_size=self.rnn_size, num_layers=self.num_layers, batch_first=True,
            bidirectional=self.bidirectional, dropout=self.dropout) # change input size
    
        
        # if self.bidirectional:
        #     self.rnn_size = self.rnn_size * 2

        self.fc2 = nn.Linear(in_features=self.rnn_size, out_features=2)
        torch.nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, input_tuple):
        # HINT: Following two methods might be useful
        # 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
        seqs, lengths = input_tuple
        
        batch_size = len(seqs)
        seqs = F.rrelu(self.drop(self.fc1(seqs)))

        seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
        _, seqs = self.gru(seqs)
        seqs = seqs[-1]

        seqs = seqs.view(batch_size, -1)
        seqs = self.fc2(seqs)
    
        return seqs