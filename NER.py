
import torch.nn as nn
import torch.nn.functional as F
import json

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()


    self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

    self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)

    self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)


def forward(self, s):
    s = self.embedding(s)

    s, _ = self.lstm(s)

    s = s.view(-1, s.shape[2])

    s = self.fc(s)

    return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    labels = labels.view(-1)

    mask = (labels >= 0).float()

    num_tokens = int(torch.sum(mask).data[0])

    outputs = outputs[range(outputs.shape[0]), labels] * mask

    return -torch.sum(outputs) / num_tokens


with jsonlines.open('input.jsonl', mode='r') as fnc:
    for i, l in enumerate(fnc.read().splitlines()):
        vocab[l] = i

train_sentences = []
train_labels = []

with jsonlines.open('input.jsonl', mode='r') as fnc:
    for sentence in fnc.read().splitlines():

        s = [vocab[token] if token in self.vocab
            else vocab['UNK']
            for token in sentence.split(' ')]
        train_sentences.append(s)

with jsonlines.open('input.jsonl', mode='r') as fnc:
    for sentence in fnc.read().splitlines():

        l = [tag_map[label] for label in sentence.split(' ')]
        train_labels.append(l)

batch_max_len = max([len(s) for s in batch_sentences])

batch_data = vocab['PAD']*np.ones((len(batch_sentences), batch_max_len))
batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

for j in range(len(batch_sentences)):
    cur_len = len(batch_sentences[j])
    batch_data[j][:cur_len] = batch_sentences[j]
    batch_labels[j][:cur_len] = batch_tags[j]

batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

train_iterator = data_iterator(train_data, params, shuffle=True)

for _ in range(num_training_steps):
    batch_sentences, batch_labels = next(train_iterator)

    output_batch = model(train_batch)

