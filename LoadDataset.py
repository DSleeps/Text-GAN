import spacy
import torch
import numpy as np

top_words = 10000

#my_tok = spacy.load('en_core_web_md')
#print(len(my_tok.vocab))

def create_vocab(file_path):
    vocab_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            for t in line.split(' '):
                if (t.find('=') != -1 or t.find('\n') != -1):
                    continue
                
                if (t.find('@') != -1):
                    t = t[1:-1]

                if (t in vocab_dict):
                    vocab_dict[t] += 1
                else:
                    vocab_dict[t] = 1

    sorted_words = list(vocab_dict.keys())
    sorted_words.sort(key=lambda x: vocab_dict[x])
    sorted_words = sorted_words[-top_words:]
    
    new_vocab = {}
    for i in range(len(sorted_words)):
        new_vocab[sorted_words[i]] = i

    return new_vocab

def sentence_split(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if (line[0] == '\n' or line[0].find('=') != -1):
                continue
            
            sentence = []
            for t in line.split(' '):
                if (t.find('@') != -1):
                    t = t[1:-1]

                if (t == '.'):
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(t)
    return sentences

def sentences_to_tensors(sentences, vocab, max_sentence_length):
    sentence_tensors = []
    sentence_lengths = []
    for sentence in sentences:
        if (len(sentence) > max_sentence_length):
            continue
        
        s_tensor = torch.zeros(max_sentence_length)
        l = 0
        for i, word in enumerate(sentence):
            if (word in vocab):
                s_tensor[i] = vocab[word]
            else:
                s_tensor[i] = vocab['<unk>']
            l = i
        sentence_tensors.append(s_tensor)
        sentence_lengths.append(l + 1)

    data = torch.zeros(len(sentence_tensors), max_sentence_length, dtype=torch.long)
    lengths = torch.zeros(len(sentence_tensors), dtype=torch.int64)
    for i, s in enumerate(sentence_tensors):
        data[i] = s
        lengths[i] = sentence_lengths[i]

    return data, lengths

# Use embed.weight.data.copy_(tensor)
def create_embedding_tensor(vocab):
    # For now I am using spacy word embeddings
    nlp = spacy.load("en_core_web_md")
    
    embedding_size = len(nlp("dog")[0].vector)
    num_words = len(vocab.keys())

    embedding_numpy = np.zeros((num_words, embedding_size))
    print('Creating embeddings...')
    for word in vocab.keys():
        token = nlp(word)
        if (len(token) != 0):
            embedding_numpy[vocab[word]] = token[0].vector
        else:
            print('Not found: ' + word)
    
    return torch.from_numpy(embedding_numpy)

def load_data(file_path, max_sentence_length):
    vocab = create_vocab('wikitext-2/wiki.train.tokens')
    sentences = sentence_split('wikitext-2/wiki.train.tokens')
    data, lengths = sentences_to_tensors(sentences, vocab, max_sentence_length)
    embedding_tensor = create_embedding_tensor(vocab)
    
    return data, lengths, vocab, embedding_tensor

#print(sentence_split('wikitext-2/wiki.train.tokens'))
