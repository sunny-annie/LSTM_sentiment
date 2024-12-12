import streamlit as st
import numpy as np
import re
import string
import torch
import torch.nn as nn
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def data_preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub('<.*?>', '', text) 
    text = ''.join([c for c in text if c not in string.punctuation])
    text = [word for word in text.split() if word not in stop_words] 
    text = ' '.join(text)
    
    return text

def padding(review_int: list, seq_len: int) -> np.array:
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)    
        
    return features

def preprocess_single_string(input_string: str, seq_len: int, vocab_to_int: dict) -> list:
    preprocessed_string = data_preprocessing(input_string)
    result_list = []
    for word in preprocessed_string.split():
        try: 
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            print(f'{e}: not in dictionary!')
    result_padded = padding([result_list], seq_len)[0]

    return torch.tensor(result_padded)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class sentimentLSTMBiDir(nn.Module):
    def __init__(self,
                vocab_size: int,
                embedding_dim: int,
                hidden_dim: int,
                n_layers: int,
                drop_prob=0.5) -> None:

        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=drop_prob,
                            batch_first=True,
                            bidirectional=True
                            )

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(hidden_dim * 2 * SEQ_LEN, 512)
        self.do = nn.Dropout()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc2(torch.tanh(self.do(self.fc1(lstm_out.flatten(1)))))
        sig_out = self.sigmoid(out)

        return sig_out

VOCAB_SIZE = 6674
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
N_LAYERS = 2
SEQ_LEN = 128

@st.cache_resource    
def load_model():
    model = sentimentLSTMBiDir(vocab_size=VOCAB_SIZE,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          n_layers=N_LAYERS)
    model.load_state_dict(torch.load('LSTMBD_model_weights.pt', map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_sentiment(review):
    model = load_model()
    vocab_to_int = torch.load('vocab.pth')
    prediction = model.to(device)(preprocess_single_string(review, seq_len=128, vocab_to_int=vocab_to_int).unsqueeze(0).to(device))
    probability = prediction[0][0]
    if probability > 0.50:
        prediction = 'Положительный отзыв'
        
    else:
        prediction = 'Отрицательный отзыв'
    return prediction