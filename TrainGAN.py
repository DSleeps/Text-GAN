import torch
from torch import nn
from AutoEncoder import EncoderDecoder
from GAN import Generator, Discriminator
from LoadDataset import load_data
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Training_parameters
g_times = 1
d_times = 1
batch_size = 32
iteration_count = 20000
max_output_len = 30

# First load the dataset
data, lengths, vocab, embedding_tensor = load_data('wikitext-103/wiki.train.tokens', max_output_len)
inv_vocab = {v: k for k, v in vocab.items()}

print(data.shape[0])

# Generator parameters
g_input_size = 100
g_hidden_size = 512
g_output_size = embedding_tensor.shape[0] # The number of words in the vocab
g_num_layers = 3
g_head_count = 5
g_learning_rate = 0.0002

generator = Generator(g_input_size, g_hidden_size, g_num_layers, g_output_size, g_head_count, device).to(device)
g_loss_fn = nn.NLLLoss()
g_optim = torch.optim.Adam(generator.parameters(), lr=g_learning_rate)

# Discriminator parameters
d_input_size = g_output_size
d_hidden_size = 256
d_num_layers = 3
d_head_count = 5
d_learning_rate = 0.0002

discriminator = Discriminator(d_input_size, d_hidden_size, d_num_layers, d_head_count, device).to(device)
d_loss_fn = nn.NLLLoss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate)

# AutoEncoder parameters
a_hidden_size = 512
a_num_layers = 2
a_file = 'AutoEncoder.t'

autoencoder = EncoderDecoder(embedding_tensor, a_hidden_size, a_num_layers, embedding_tensor.shape[0], device).to(device)
autoencoder.load_state_dict(torch.load(a_file))

def make_sentence(output):
    maxes = torch.argmax(output, dim=1)
    sentence = ''
    for w in maxes:
        sentence += inv_vocab[w.item()] + ' '
    return sentence

# This function takes outputs from the generator and the discriminator and returns the 
# discriminator outputs that correspond to the eos tag of the generator inputs. This 
# ensures it doesn't look past the periods
def get_d_output(g_out, d_out):
    # Converts it into the indices
    i_output = torch.argmax(g_out, dim=2)
    eos_index = vocab['.']
    
    # Find where all of the 'eos' tokens are
    eos_indices = (i_output == eos_index).nonzero()
    
    # Here's a bunch of hacky stuff to get what I want
    temp = torch.zeros(batch_size, max_output_len, dtype=torch.long).to(device)
    temp[:] = max_output_len-1
    temp[eos_indices[:,0],eos_indices[:,1]] = eos_indices[:,1]
    
    first_eos_indices = torch.min(temp, dim=1)
    return d_out[torch.arange(0,batch_size),first_eos_indices.values]


# Start the training loop
start_time = time.time()
for i in range(iteration_count):
    for d in range(d_times):
        # First generate the autoencoder outputs
        choices = np.random.choice(data.shape[0], int(batch_size/2))
        seqs = data[choices].to(device)
        s_lengths = lengths[choices].to(device)

        auto_outs = autoencoder(seqs, s_lengths, max_output_len)
        
        # Now generate the generator outputs
        input_seed = torch.empty(int(batch_size/2), g_input_size).normal_(mean=0.5, std=1.0).to(device)
        inputs = torch.zeros(max_output_len, int(batch_size/2), g_input_size).to(device)
        inputs[:] = input_seed
        g_outs = generator(inputs.permute(1,0,2), max_output_len)
        
        d_inputs = torch.cat((auto_outs, g_outs), dim=0)
        d_desired = torch.zeros(batch_size,dtype=torch.int64).to(device)
        d_desired[:int(batch_size/2)] = 0   # The first index is real
        d_desired[int(batch_size/2):] = 1   # The second index is fake inputs
        
        d_outs = discriminator(d_inputs, max_output_len)
        d_loss = d_loss_fn(get_d_output(d_inputs, d_outs), d_desired)
        # d_loss = d_loss_fn(d_outs[:,-1], d_desired)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
       
    out_example = None
    for g in range(g_times):
        # Generator outputs
        input_seed = torch.empty(batch_size, g_input_size).normal_(mean=0.5, std=1.0).to(device)
        inputs = torch.zeros(max_output_len, batch_size, g_input_size).to(device)
        inputs[:] = input_seed
        
        g_outs = generator(inputs.permute(1,0,2), max_output_len)
        out_example = g_outs[-1]

        d_outs = discriminator(g_outs, max_output_len)

        d_desired = torch.zeros(batch_size, dtype=torch.int64).to(device)
        d_desired[:batch_size] = 0   # The first index is real and the generator wants the
                                       # discriminator to think everything it outputs is real
        g_loss = g_loss_fn(get_d_output(g_outs, d_outs), d_desired)
        # g_loss = g_loss_fn(d_outs[:,-1], d_desired)

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

    if (i % 1000 == 0):
        # Calculate the time remaining
        secs_per_iter = (time.time() - start_time)/(i + 1)
        remaining_mins = int((secs_per_iter/60.) * (iteration_count - i))
        print('Iteration ' + str(i) + ' | Time Remaining: ' + str(remaining_mins) + ' mins')
        print(g_loss)
        print(d_loss)
        print(make_sentence(out_example))
