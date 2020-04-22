import torch
from AutoEncoder import EncoderDecoder
from LoadDataset import load_data
from GAN import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
'''
hidden_size = 10
num_layers = 2
output_size = 20

batch_size = 5
max_output_len = 20
max_seq_len = 40

data, lengths, vocab, embedding_tensor = load_data('wikitext-2/wiki.test.tokens', max_seq_len)

model = EncoderDecoder(embedding_tensor, hidden_size, num_layers, output_size)

#lengths = torch.randint(batch_size, max_seq_len, (batch_size,))
#sequences = torch.randn(batch_size, max_seq_len, input_size)
print(data.shape)
print(model(data, lengths, max_output_len))
'''

'''
input_size = 30
hidden_size = 50
num_layers = 2
output_size = 10
head_count = 5
batch_size = 4
max_output_len = 30

model = Generator(input_size, hidden_size, num_layers, output_size, head_count, device).to(device)

inputs = torch.randn(batch_size, max_output_len, input_size).to(device)

print(model(inputs, max_output_len))
'''

batch_size = 20
max_output_len = 30
def get_d_output(g_out, d_out):
    # Converts it into the indices
    i_output = torch.argmax(g_out, dim=2)
    eos_index = 5

    # Find where all of the 'eos' tokens are
    eos_indices = (i_output == eos_index).nonzero()

    # Here's a bunch of hacky stuff to get what I want
    temp = torch.zeros(batch_size, max_output_len, dtype=torch.long).to(device)
    temp[:] = max_output_len-1
    temp[eos_indices[:,0],eos_indices[:,1]] = eos_indices[:,1]

    first_eos_indices = torch.min(temp, dim=1)
    
    return d_out[torch.arange(0,batch_size),first_eos_indices.values]

d_outs = torch.randn(batch_size, max_output_len, 2).to(device)
g_outs = torch.randn(batch_size, max_output_len, 6).to(device)
print(get_d_output(g_outs, d_outs))
