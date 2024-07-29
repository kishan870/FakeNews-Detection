import pandas as pd
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import pickle

import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.embedding(x)
            x, (hn, cn) = self.lstm(x)
            x = self.fc(x[:, -1, :])
            x = self.sigmoid(x)
            return x
        
def LoadModel():
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    word_index = tokenizer.word_index
    input_dim = len(word_index) + 1
    hidden_dim = 128
    output_dim = 1

    best_model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    return best_model



