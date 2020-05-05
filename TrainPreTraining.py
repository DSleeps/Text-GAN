import torch
from torch import nn
from NewAutoEncoder import EncoderDecoder
from NewGAN import Generator, Discriminator, GenEncoderDecoder
from LoadDataset import load_data
import numpy as np
import time
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Training_parameters
g_times = 3
d_times = 2
a_times = 1
batch_size = 32
pre_training_iteration_count = 15000
d_pre_training_iteration_count = 200
iteration_count = 10000
max_output_len = 30

# First load the dataset
data, lengths, vocab, embedding_tensor = load_data('wikitext-103/wiki.train.tokens', max_output_len)
inv_vocab = {v: k for k, v in vocab.items()}

print(data.shape[0])

# Generator parameters
g_input_size = 300
g_hidden_size = 512
g_output_size = 300 # embedding_tensor.shape[0] # The number of words in the vocab
g_num_layers = 2
g_head_count = 3
g_learning_rate = 0.0001

generator = GenEncoderDecoder(g_input_size, g_hidden_size, g_num_layers, g_output_size, g_head_count, device, use_output=True).to(device)
g_loss_fn = nn.BCELoss()
# g_optim = torch.optim.SGD(generator.parameters(), lr=g_learning_rate, momentum=0.9)
g_optim = torch.optim.Adam(generator.parameters(), lr=g_learning_rate)

# Discriminator parameters
d_input_size = g_output_size
d_hidden_size = 512
d_num_layers = 2
d_head_count = 3
d_learning_rate = 0.0001

discriminator = Discriminator(d_input_size, d_hidden_size, d_num_layers, d_head_count, device).to(device)
d_loss_fn = nn.BCELoss()
# d_optim = torch.optim.SGD(discriminator.parameters(), lr=d_learning_rate, momentum=0.9)
d_optim = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate)

# AutoEncoder parameters
a_hidden_size = g_output_size
a_num_layers = 2
a_learning_rate = 0.0005
a_file = 'AutoEncoder.t'

autoencoder = EncoderDecoder(embedding_tensor, a_hidden_size, a_num_layers, embedding_tensor.shape[0], device).to(device)
a_loss_fn = nn.NLLLoss()
a_optim = torch.optim.Adam(autoencoder.parameters(), lr=a_learning_rate)
autoencoder.load_state_dict(torch.load(a_file))

def make_sentence(output):
    maxes = torch.argmax(autoencoder.decoder.output_model(output), dim=1)
    sentence = ''
    for w in maxes:
        sentence += inv_vocab[w.item()] + ' '
    return sentence

def seq_to_sentence(seq):
    sentence = ''
    for w in seq:
        sentence += inv_vocab[w.item()] + ' '
    return sentence

# This function takes outputs from the generator and the discriminator and returns the 
# discriminator outputs that correspond to the eos tag of the generator inputs. This 
# ensures it doesn't look past the periods
def get_d_output(g_out, d_out):
    # Converts it into the indices
    i_output = torch.argmax(autoencoder.decoder.output_model(g_out), dim=2)
    eos_index = vocab['.']
    
    # Find where all of the 'eos' tokens are
    eos_indices = (i_output == eos_index).nonzero()
    
    # Here's a bunch of hacky stuff to get what I want
    temp = torch.zeros(batch_size, max_output_len, dtype=torch.long).to(device)
    temp[:] = max_output_len-1
    temp[eos_indices[:,0],eos_indices[:,1]] = eos_indices[:,1]
     
    first_eos_indices = torch.min(temp, dim=1)
    return d_out[torch.arange(0,batch_size),first_eos_indices.values]

def calculate_accuracy(d_outs, d_desired):
    outs = (d_outs > 0.5).float()
    same = (outs == d_desired).sum()

    return float(same)/d_outs.shape[0]

print('Auto done')

# First start by pretraining the generator
PAD_INDEX = 0
clear_percent = 0.2
pre_loss = nn.NLLLoss(ignore_index=PAD_INDEX)
for i in range(pre_training_iteration_count):
    choices = np.random.choice(data.shape[0], int(batch_size))
    seqs = data[choices].to(device)
    s_lengths = lengths[choices].to(device)
    
    inputs = autoencoder.encoder.embedding(seqs)
    
    # Now clear out a random number of words
    temp = torch.ones(batch_size * max_output_len).to(device)
    rand_clear = np.random.choice(batch_size * max_output_len, int(clear_percent*(batch_size * max_output_len)))
    temp[rand_clear] = 0
    temp = torch.reshape(temp, (batch_size, max_output_len, 1))
    inputs = inputs * temp

    g_outs = generator(inputs, s_lengths, max_output_len)
    o_outs = autoencoder.decoder.output_model(g_outs)
    
    loss = pre_loss(o_outs.permute(0,2,1), seqs)
    if (i % 200 == 0):
        # Calculate the time remaining
        print('Iteration ' + str(i))
        print(loss)
        print(make_sentence(g_outs[-1]))
        print(seq_to_sentence(seqs[-1]))
        print(seq_to_sentence(seqs[-1] * torch.reshape(temp[-1], (max_output_len,))))
    
    g_optim.zero_grad()
    loss.backward()
    g_optim.step()

# Pre train the discriminator a bit
for d in range(d_pre_training_iteration_count):
    # First generate the autoencoder outputs
    choices = np.random.choice(data.shape[0], int(batch_size/2))
    seqs = data[choices].to(device)
    s_lengths = lengths[choices].to(device)

    auto_outs, a_hidden = autoencoder(seqs, s_lengths, max_output_len) 
    random_noise = torch.empty(int(batch_size/2), max_output_len, a_hidden_size).normal_(mean=0, std=0.2).to(device)
    a_hidden += random_noise
    a_example = a_hidden[-1]
    # Now generate the generator outputs
    # input_seed = torch.empty(int(batch_size/2), g_input_size).normal_(mean=0, std=1.0).to(device)
    # inputs = torch.zeros(max_output_len, int(batch_size/2), g_input_size).to(device)
    # inputs[:] = input_seed
    # inputs = inputs.permute(1,0,2)
    # g_outs = generator(inputs.permute(1,0,2), max_output_len)
    
    inputs = torch.empty(int(batch_size/2), max_output_len, g_input_size).normal_(mean=0, std=1.0).to(device)
    rand_ends = torch.randint(10, max_output_len, (int(batch_size/2),)).to(device)
    m_rand_ends = rand_ends + torch.arange(int(batch_size/2)).to(device) * max_output_len
    mask = torch.arange(int(batch_size/2)*max_output_len).view(int(batch_size/2),max_output_len).to(device)
    mask = mask < m_rand_ends.view(int(batch_size/2),1)
    inputs = inputs * mask.view(int(batch_size/2), max_output_len, 1)

    g_outs = generator(inputs, rand_ends, max_output_len)

    d_inputs = torch.cat((a_hidden, g_outs), dim=0)
    d_desired = torch.zeros(batch_size).to(device)
    d_desired[:int(batch_size/2)] = 1   # The second index is real
    d_desired[int(batch_size/2):] = 0   # The first index is fake inputs
    
    d_outs = discriminator(d_inputs, max_output_len)
    d_outs = get_d_output(d_inputs, d_outs).view(batch_size)
    d_loss = d_loss_fn(d_outs, d_desired)
    # d_loss = d_loss_fn(d_outs[:,-1], d_desired)
    
    if (float(d_loss) < 0.1):
        break

    d_optim.zero_grad()
    d_loss.backward()
    d_optim.step()
 

# Start the training loop
d_losses = []
a_d_loss = 0
g_losses = []
a_g_loss = 0
start_time = time.time()
for i in range(iteration_count): 
    a_example = None
    for d in range(d_times):
        discriminator.zero_grad()
        # First generate the autoencoder outputs
        choices = np.random.choice(data.shape[0], int(batch_size/2))
        seqs = data[choices].to(device)
        s_lengths = lengths[choices].to(device)

        auto_outs, a_hidden = autoencoder(seqs, s_lengths, max_output_len) 
        random_noise = torch.empty(int(batch_size/2), max_output_len, a_hidden_size).normal_(mean=0, std=0.2).to(device)
        a_hidden += random_noise
        a_example = a_hidden[-1]
        # Now generate the generator outputs
        # input_seed = torch.empty(int(batch_size/2), g_input_size).normal_(mean=0, std=1.0).to(device)
        # inputs = torch.zeros(max_output_len, int(batch_size/2), g_input_size).to(device)
        # inputs[:] = input_seed
        # inputs = inputs.permute(1,0,2)
        # g_outs = generator(inputs.permute(1,0,2), max_output_len)
        
        inputs = torch.empty(int(batch_size/2), max_output_len, g_input_size).normal_(mean=0, std=1.0).to(device)
        rand_ends = torch.randint(10, max_output_len, (int(batch_size/2),)).to(device)
        m_rand_ends = rand_ends + torch.arange(int(batch_size/2)).to(device) * max_output_len
        mask = torch.arange(int(batch_size/2)*max_output_len).view(int(batch_size/2),max_output_len).to(device)
        mask = mask < m_rand_ends.view(int(batch_size/2),1)
        inputs = inputs * mask.view(int(batch_size/2), max_output_len, 1)

        g_outs = generator(inputs, rand_ends, max_output_len)

        d_inputs = torch.cat((a_hidden, g_outs), dim=0)
        d_desired = torch.zeros(batch_size).to(device)
        d_desired[:int(batch_size/2)] = 1   # The second index is real
        d_desired[int(batch_size/2):] = 0   # The first index is fake inputs
        
        d_outs = discriminator(d_inputs, max_output_len)
        d_outs = get_d_output(d_inputs, d_outs).view(batch_size)
        d_loss = d_loss_fn(d_outs, d_desired)

        a_d_loss += float(d_loss)
        # d_loss = d_loss_fn(d_outs[:,-1], d_desired)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
       
    out_example = None
    for g in range(g_times):
        generator.zero_grad()
        # Generator outputs
        # input_seed = torch.empty(batch_size, g_input_size).normal_(mean=0, std=1.0).to(device)
        # inputs = torch.zeros(max_output_len, batch_size, g_input_size).to(device)
        # inputs[:] = input_seed
        # inputs = inputs.permute(1,0,2)
        # g_outs = generator(inputs.permute(1,0,2), max_output_len)
        
        inputs = torch.empty(batch_size, max_output_len, g_input_size).normal_(mean=0, std=1.0).to(device)
        rand_ends = torch.randint(10, max_output_len, (batch_size,)).to(device)
        m_rand_ends = rand_ends + torch.arange(batch_size).to(device) * max_output_len
        mask = torch.arange(batch_size*max_output_len).view(batch_size,max_output_len).to(device)
        mask = mask < m_rand_ends.view(batch_size,1)
        inputs = inputs * mask.view(batch_size, max_output_len, 1)
        
        g_outs = generator(inputs, rand_ends, max_output_len)
        out_example = g_outs[-1]

        d_outs = discriminator(g_outs, max_output_len)
        d_outs = get_d_output(g_outs, d_outs).view(batch_size)
        d_desired = torch.zeros(batch_size).to(device)
        d_desired[:batch_size] = 1   # The first second is real and the generator wants the
                                     # discriminator to think everything it outputs is real
        g_loss = g_loss_fn(d_outs, d_desired)
        a_g_loss += float(g_loss)
        # g_loss = g_loss_fn(d_outs[:,-1], d_desired)

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
        
        # d_accuracy = 1 - calculate_accuracy(d_outs, d_desired)
        # print(d_accuracy)
        # print(d_outs)
        # print(g_loss)
        # if (d_accuracy < 0.5):
        #   break

    if (i % 200 == 0):
        # Calculate the time remaining
        secs_per_iter = (time.time() - start_time)/(i + 1)
        remaining_mins = int((secs_per_iter/60.) * (iteration_count - i))
        print('Iteration ' + str(i) + ' | Time Remaining: ' + str(remaining_mins) + ' mins')
        print(a_g_loss/600.)
        print(a_d_loss/400.)
        print(make_sentence(out_example))
        print(make_sentence(a_example))
        
        if (i != 0):
            d_losses.append(a_d_loss/400.)
            g_losses.append(a_g_loss/600.)
        else:
            d_losses.append(a_d_loss/2.)
            g_losses.append(a_g_loss/3.)
        a_d_loss = 0
        a_g_loss = 0

plt.plot(g_losses)
plt.plot(d_losses)
plt.show()
