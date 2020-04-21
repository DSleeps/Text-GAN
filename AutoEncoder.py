import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# This is what you have to do for the loss. Set the proper ignore index
# criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)

class Encoder(nn.Module):

    def __init__(self, embedding_tensor, hidden_size, num_layers, device):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_tensor.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.embedding = nn.Embedding(embedding_tensor.shape[0], embedding_tensor.shape[1])
        self.embedding.load_state_dict({'weight': embedding_tensor})
        self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(embedding_tensor.shape[1], hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, inputs, lengths):
        batch_size = inputs.shape[0]
        max_sequence_len = inputs.shape[1]

        # We want to use word embeddings so use these pretrained ones
        inputs = self.embedding(inputs)
        
        # First pad the input sequence correctly
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

        # We are going to want to implement attention so return the whole thing
        outputs, hiddens = self.rnn(packed_inputs)
        outputs = pad_packed_sequence(outputs, batch_first=True, total_length=max_sequence_len)[0]
        return outputs, hiddens

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        self.rnn = nn.GRU(input_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(2 * hidden_size,1),
                                       nn.ReLU())
        
        self.output_model = nn.Sequential(nn.Linear(2 * hidden_size, output_size),
                                          nn.LogSoftmax())

    def forward(self, inputs, lengths, final_hidden, hiddens, max_output_len, max_sequence_len):
        batch_size = inputs.shape[0] 
        outputs = torch.zeros(batch_size, max_output_len, self.output_size).to(self.device)
        
        # Make the mask
        r_mask = torch.ones(batch_size, max_sequence_len).to(self.device)
        for i in range(batch_size):
            r_mask[i][lengths[i]:] = 0
        r_mask = torch.reshape(r_mask, (batch_size, max_sequence_len, 1))

        prev_hidden = final_hidden.permute(1,0,2)
        for i in range(max_output_len):
            # Reformat the hidden layer to input into the attention model
            last_hidden =  prev_hidden[:,-1,:]
            dup_hidden = torch.zeros(max_sequence_len, batch_size, self.hidden_size).to(self.device)
            dup_hidden[:] = last_hidden
            dup_hidden = dup_hidden.permute(1,0,2) * r_mask

            # First calculate the attention
            attention = self.attention(torch.cat((dup_hidden, hiddens), dim=2))
            # attention = torch.reshape(attention, (batch_size, max_sequence_len))
            attention = nn.Softmax(dim = 1)(attention)
            c_vecs = torch.sum(attention * hiddens, dim=1)
            c_vecs = torch.reshape(c_vecs, (batch_size, 1, self.hidden_size))
            
            # Now feed that into the model
            output, hidden = self.rnn(torch.cat((inputs[:,i:i+1,:], c_vecs), dim=2), prev_hidden.permute(1,0,2))
            prev_hidden = hidden.permute(1,0,2)
            
            # Run the hidden layer through the output model
            outputs[:,i] = self.output_model(torch.cat((output[:,0,:], c_vecs[:,0,:]), dim=1))

        return outputs

class EncoderDecoder(nn.Module):

    def __init__(self, embedding_tensor, hidden_size, num_layers, output_size, device):
        super(EncoderDecoder, self).__init__()

        self.input_size = embedding_tensor.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.encoder = Encoder(embedding_tensor, hidden_size, num_layers, device)
        self.decoder = Decoder(self.input_size, hidden_size, num_layers, output_size, device)

    def forward(self, inputs, lengths, max_output_len):
        batch_size = inputs.shape[0]

        hiddens, final_hidden = self.encoder(inputs, lengths)
        outputs = self.decoder(torch.zeros(batch_size, max_output_len, self.input_size).to(self.device), lengths, final_hidden, hiddens, max_output_len, inputs.shape[1])
        
        # Here I might need to truncate the responses after the sentence end character
        return outputs

