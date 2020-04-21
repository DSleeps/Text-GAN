import torch
from torch import nn
from AutoEncoder import EncoderDecoder
from GAN import Generator, Discriminator
from LoadDataset import load_data
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Training_parameters
g_times = 1
d_times = 5
batch_size = 32
iteration_count = 10000
max_output_len = 30

# First load the dataset
data, lengths, vocab, embedding_tensor = load_data('wikitext-2/wiki.train.tokens', max_output_len)

# Generator parameters
g_input_size = 100
g_hidden_size = 256
g_output_size = embedding_tensor.shape[0] # The number of words in the vocab
g_num_layers = 3
g_head_count = 5
g_learning_rate = 0.005

generator = Generator(g_input_size, g_hidden_size, g_num_layers, g_output_size, g_head_count, device).to(device)
g_loss_fn = nn.NLLLoss()
g_optim = torch.optim.Adam(generator.parameters(), lr=g_learning_rate)

# Discriminator parameters
d_input_size = g_output_size
d_hidden_size = 256
d_num_layers = 3
d_head_count = 5
d_learning_rate = 0.0005

discriminator = Discriminator(d_input_size, d_hidden_size, d_num_layers, d_head_count, device).to(device)
d_loss_fn = nn.NLLLoss()
d_optim = torch.optim.Adam(discriminator.parameters(), lr=d_learning_rate)


# AutoEncoder parameters
a_hidden_size = 256
a_num_layers = 2
a_file = 'Autoencoder.t'

autoencoder = EncoderDecoder(embedding_tensor, a_hidden_size, a_num_layers, embedding_tensor.shape[0], device).to(device)
autoencoder.load_state_dict(torch.load(a_file))
autoencoder.eval()

# Start the training loop
for i in range(iteration_count):
    generator.eval()
    discriminator.train()
    for d in d_times:
        # First generate the autoencoder outputs
        choices = np.random.choice(data.shape[0], batch_size/2)
        seqs = data[choices].to(device)
        s_lengths = lengths[choices].to(device)

        auto_outs = autoencoder(seqs, s_lengths, max_output_len)

        # Now generate the generator outputs
        input_seed = torch.empty(g_input_size).normal_(mean=0, std=1.0).to(device)
        inputs = torch.zeros(batch_size/2, max_output_len, g_input_size).to(device)
        inputs[:] = input_seed
        g_outs = generator(inputs, max_output_len)

        d_inputs = torch.cat((auto_outs, g_outs), dim=0)
        d_desired = torch.zeros(batch_size,1)
        d_desired[:batch_size/2,0] = 0   # The first index is real
        d_desired[batch_size/2:,0] = 1   # The second index is fake inputs

        d_outs = discriminator(d_inputs, max_output_len)

        d_loss = d_loss_fn(d_outs.permute(0,2,1), d_desired)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
    generator.train()
    discriminator.eval()
    for g in g_times:
        # Generator outputs
        input_seed = torch.empty(g_input_size).normal_(mean=0, std=1.0).to(device)
        inputs = torch.zeros(batch_size/2, max_output_len, g_input_size).to(device)
        inputs[:] = input_seed

        g_outs = generator(inputs, max_output_len)
        d_outs = discriminator(g_outs, max_output_len)

        d_desired = torch.zeros(batch_size,1)
        d_desired[:batch_size,0] = 0   # The first index is real and the generator wants the
                                       # discriminator to think everything it outputs is real
        g_loss = loss_fn(d_outs, d_desired)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimier.step()

    

