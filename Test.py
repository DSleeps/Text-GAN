import torch
from AutoEncoder import EncoderDecoder
from LoadDataset import load_data

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
