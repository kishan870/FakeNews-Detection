from flask import Flask, request, render_template, jsonify

import pandas as pd
import nltk
from nltk.corpus import stopwords

import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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
        

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

word_index = tokenizer.word_index
input_dim = 5000
hidden_dim = 128
output_dim = 1

print('INPUT DIM: ', input_dim)

best_model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
best_model.load_state_dict(torch.load('best_model.pth', map_location=device))
best_model.eval()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)
    
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    processed_text = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    features_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)
    prediction = best_model(features_tensor).item()

    return jsonify({'prediction': 'Real' if prediction > 0.5 else 'Fake'})

if __name__ == '__main__':
    app.run(debug=True)



    


