import torch
from AutoEncoder import EncoderDecoder
from LoadDataset import load_data
from GAN import Generator
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

input_size = 30
hidden_size = 50
num_layers = 2
output_size = 10
head_count = 5
batch_size = 4
max_output_len = 30

model = Generator(input_size, hidden_size, num_layers, output_size, head_count, 'cpu')

inputs = torch.randn(batch_size, max_output_len, input_size)

print(model(inputs, max_output_len))
