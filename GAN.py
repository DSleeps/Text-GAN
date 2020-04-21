import torch
from torch import nn
import numpy as np

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, head_count, device):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.head_count = head_count
        self.device = device

        self.lstm = nn.LSTM(input_size + hidden_size * head_count, hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, head_count),
                                       nn.ReLU())

        self.output_model = nn.Sequential(nn.Linear((1 + head_count) * hidden_size, output_size),
                                          nn.LogSoftmax())

    def forward(self, inputs, max_output_len):
        batch_size = inputs.shape[0]
        outputs = torch.zeros(batch_size, max_output_len, self.output_size).to(self.device)

        prev_hidden = torch.zeros(batch_size, self.num_layers, self.hidden_size)
        prev_cell = torch.zeros(batch_size, self.num_layers, self.hidden_size)
        hiddens = torch.zeros(batch_size, 1, self.hidden_size)
        for i in range(max_output_len):
            # Reformat the hidden layer to input into the attention model
            last_hidden =  prev_hidden[:,-1,:]
            dup_hidden = torch.zeros(i+1, batch_size, self.hidden_size).to(self.device)
            dup_hidden[:] = last_hidden
            dup_hidden = dup_hidden.permute(1,0,2)

            # First calculate the attention
            attention = self.attention(torch.cat((dup_hidden, hiddens), dim=2))
            # attention = torch.reshape(attention, (batch_size, max_sequence_len))
            attention = nn.Softmax(dim = 1)(attention)
            
            # Reshape them for the matrix multiply
            attention = torch.reshape(attention, (batch_size, i+1, self.head_count, 1))
            r_hiddens = torch.reshape(hiddens, (batch_size, i+1, 1, self.hidden_size))

            c_vecs = torch.sum(attention * r_hiddens, dim=1)
            c_vecs = torch.reshape(c_vecs, (batch_size, 1, self.hidden_size * self.head_count))
            
            # Now feed that into the model
            output, outs = self.lstm(torch.cat((inputs[:,i:i+1,:], c_vecs), dim=2), (prev_hidden.permute(1,0,2), prev_cell.permute(1,0,2)))
            
            prev_hidden = outs[0].permute(1,0,2)
            prev_cell = outs[1].permute(1,0,2)
            hiddens = torch.cat((hiddens, output), dim=1) 

            # Run the hidden layer through the output model
            outputs[:,i] = self.output_model(torch.cat((output[:,0,:], c_vecs[:,0,:]), dim=1))

        return outputs

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, head_count, device):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = 2
        self.head_count = head_count
        self.device = device

        self.lstm = nn.LSTM(input_size + hidden_size * head_count, hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, head_count),
                                       nn.ReLU())

        self.output_model = nn.Sequential(nn.Linear((1 + head_count) * hidden_size, 2),
                                          nn.LogSoftmax())

    def forward(self, inputs, max_output_len):
        batch_size = inputs.shape[0]
        outputs = torch.zeros(batch_size, max_output_len, self.output_size).to(self.device)

        prev_hidden = torch.zeros(batch_size, self.num_layers, self.hidden_size)
        prev_cell = torch.zeros(batch_size, self.num_layers, self.hidden_size)
        hiddens = torch.zeros(batch_size, 1, self.hidden_size)
        for i in range(max_output_len):
            # Reformat the hidden layer to input into the attention model
            last_hidden =  prev_hidden[:,-1,:]
            dup_hidden = torch.zeros(i+1, batch_size, self.hidden_size).to(self.device)
            dup_hidden[:] = last_hidden
            dup_hidden = dup_hidden.permute(1,0,2)

            # First calculate the attention
            attention = self.attention(torch.cat((dup_hidden, hiddens), dim=2))
            # attention = torch.reshape(attention, (batch_size, max_sequence_len))
            attention = nn.Softmax(dim = 1)(attention)
            
            # Reshape them for the matrix multiply
            attention = torch.reshape(attention, (batch_size, i+1, self.head_count, 1))
            r_hiddens = torch.reshape(hiddens, (batch_size, i+1, 1, self.hidden_size))

            c_vecs = torch.sum(attention * r_hiddens, dim=1)
            c_vecs = torch.reshape(c_vecs, (batch_size, 1, self.hidden_size * self.head_count))
            
            # Now feed that into the model
            output, outs = self.lstm(torch.cat((inputs[:,i:i+1,:], c_vecs), dim=2), (prev_hidden.permute(1,0,2), prev_cell.permute(1,0,2)))
            
            prev_hidden = outs[0].permute(1,0,2)
            prev_cell = outs[1].permute(1,0,2)
            hiddens = torch.cat((hiddens, output), dim=1) 

            # Run the hidden layer through the output model
            outputs[:,i] = self.output_model(torch.cat((output[:,0,:], c_vecs[:,0,:]), dim=1))

        return outputs[:,-1]
