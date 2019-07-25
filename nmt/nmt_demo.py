#!/usr/bin/python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: nmt_demo.py
# Author: Junyi Li
# Personal page: dukeenglish.github.io
# Created Time: 09:43:54 2019-07-25
#########################################################################
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: "<SOS>", 1: "<EOS>", -1: "<unk>"}
        self.idx = 2 # Count SOS and EOS

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return -1
        return self.word2idx[word]

    def __len__(self):
        return self.idx

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, seq_input, hidden):
        embedded = self.embedding(seq_input).view(1, 1, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return hidden

    def sample(self,seq_list):
        word_inds = torch.LongTensor(seq_list).to(device)
        h = self.initHidden()
        for word_tensor in word_inds:
            h = self(word_tensor,h)
        return h

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 10

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq_input, hidden):
        output = self.embedding(seq_input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def sample(self,pre_hidden):
        inputs = torch.tensor([SOS_token], device=device)
        hidden = pre_hidden
        res = [SOS_token]
        for i in range(self.maxlen):
            output,hidden = self(inputs,hidden)
            topv, topi = output.topk(1)
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            inputs = topi.squeeze().detach()
        return res


lan1 = Vocabulary()
lan2 = Vocabulary()

data = [['你 很 聪明 。', 'you are very wise .'],
        ['我们 一起 打 游戏 。', 'let us play game together .'],
        ['你 太 刻薄 了 。', 'you are so mean .'],
        ['你 完全 正确 。', 'you are perfectly right .'],
        ['我 坚决 反对 妥协 。', 'i am strongly opposed to a compromise .'],
        ['他们 正在 看 电影 。', 'they are watching a movie .'],
        ['他 正在 看着 你 。', 'he is looking at you .'],
        ['我 怀疑 他 是否 会 来', 'i am doubtful whether he will come .']]

for i,j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)

def sentence2tensor(lang, sentence):
    indexes = [lang(word) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def pair2tensor(pair):
    input_tensor = sentence2tensor(lan1, pair[0])
    target_tensor = sentence2tensor(lan2, pair[1])
    return (input_tensor, target_tensor)

learning_rate = 0.001
hidden_size = 256

encoder = EncoderRNN(len(lan1),hidden_size).to(device)
decoder = DecoderRNN(hidden_size,len(lan2)).to(device)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

loss = 0
criterion = nn.NLLLoss()

turns = 200

print_every = 20
print_loss_total = 0
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

for turn in range(turns):
    optimizer.zero_grad()
    loss = 0

    x,y = training_pairs[turn]
    input_length = x.size(0)
    target_length = y.size(0)

    h = encoder.initHidden()
    for i in range(input_length):
        h = encoder(x[i],h)

    decoder_input = torch.LongTensor([SOS_token]).to(device)

    for i in range(target_length):
        decoder_output,h = decoder(decoder_input,h)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, y[i])
        if decoder_input.item() == EOS_token:break

    print_loss_total += loss.item()/target_length
    if (turn+1) % print_every == 0 :
        print("loss:{loss:,.4f}".format(loss=print_loss_total/print_every))
        print_loss_total = 0

    loss.backward()
    optimizer.step()

def translate(s):
    t = [lan1(i) for i in s.split()]
    t.append(EOS_token)
    f = encoder.sample(t)
    s = decoder.sample(f)
    r = [lan2.idx2word[i] for i in s]
    return ' '.join(r)

for pr in data:
    print('>>',pr[0])
    print('==',pr[1])
    print('result:',translate(pr[0]))
    print()

