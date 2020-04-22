import torch
from torch import nn
from AutoEncoder import EncoderDecoder
from LoadDataset import load_data
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

hidden_size = 256
num_layers = 2

iteration_count = 10000
batch_size = 32
learning_rate = 0.0005
max_seq_len = 30
max_output_len = 30

# Load the dataset
data, lengths, vocab, embedding_tensor = load_data('wikitext-2/wiki.train.tokens', max_seq_len)
inv_vocab = {v: k for k, v in vocab.items()}

# Create the model
model = EncoderDecoder(embedding_tensor, hidden_size, num_layers, embedding_tensor.shape[0], device).to(device)

# Initialize the loss and optimizer
PAD_INDEX = 0
loss_fn = nn.NLLLoss(ignore_index=PAD_INDEX)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def make_sentence(output):
    maxes = torch.argmax(output, dim=1)
    sentence = ''
    for w in maxes:
        sentence += inv_vocab[w.item()] + ' '
    return sentence

def convert_seq(seq):
    sentence = ''
    for w in seq:
        sentence += inv_vocab[w.item()] + ' '
        print(w.item())
    return sentence

for i in range(iteration_count):
    # Pick a random batch
    choices = np.random.choice(data.shape[0], batch_size)

    seqs = data[choices].to(device)
    s_lengths = lengths[choices].to(device)

    outputs = model(seqs, s_lengths, max_output_len)
    loss = loss_fn(outputs.permute(0,2,1), seqs)
    
    if (i % 200 == 0):
        print('Iteration ' + str(i))
        print(loss)
        print(make_sentence(outputs[-1]))
        print(convert_seq(seqs[-1]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'AutoEncoder.t')
