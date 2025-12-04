import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from time2vec import Time2Vec



class LSTM_EMB_T(nn.Module):
    def __init__(self, sen_embedding_size=384, lstm_hidden_size=64, time_embedding_size=64, lstm_layer = 1):
        super(LSTM_EMB_T, self).__init__()

        self.embedding_dropout = nn.Dropout(p=0.5)
        self.time2vec = Time2Vec(time_embedding_size) # time t -> [time_embedding_size]
        
        vector_size = time_embedding_size+sen_embedding_size

        self.lstm = nn.LSTM(
            input_size=vector_size, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layer,
            bidirectional=False
        )

        self.fc = nn.Linear(lstm_hidden_size,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, t, h=None):

        t_emb = self.time2vec(t)
        x = torch.cat([x, t_emb], dim=-1)
        x = self.embedding_dropout(x) # dropout for both time and emb

        x, h = self.lstm(x, h)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out, h

# # Learnable Embeddings for LSTM
# # NOT COMPLETE + Mistakes(DO NOT USE)
# # Try to COMPLETE 
#       IF your are able to pad input data hierchically(word level) then sentence level to match shape for input
#        (batch, max word per batch, max sentence per batch) and LSTM to ignore those paddings
# class LSTM_HE(nn.Module):
#     def __init__(self, nwords, word_embed_size, sen_embedding_size, hidden_size):
#         super(LSTM_HE, self).__init__()
#         # embeddings for the sentence words 
#         # for each word calculate its embeddings (n, D)
#         self.embedding = nn.Embedding(nwords, word_embed_size)   # ->  (n, D)
#         self.embedding_dropout = nn.Dropout(p=0.5)               # dropout regularizer
#         self.sen_embedding = nn.LSTM(
#             input_size=word_embed_size, 
#             hidden_size=sen_embedding_size,
#             num_layers=1,
#             bidirectional=False
#         )
        
#         # Process each sequence one by one
#         self.rnn = nn.LSTM(
#             input_size=sen_embedding_size, 
#             hidden_size=hidden_size,
#             num_layers=1,
#             bidirectional=False
#         )

#         # Binary classification
#         self.fc = nn.Linear(hidden_size,1)
#         self.sigmoid = nn.Sigmoid()  # or comment out to just keep logits (no sigmoid)
        
        

#     def forward(self, x, h=None):
#         x = self.embedding(x)
#         x = self.embedding_dropout(x)
#         print("word embedding: \n",x)
#         print()
#         _, (h_n, c_n) = self.sen_embedding(x)   # x is the sequence of outputs, h_n, c_n => final hidden, cell state
#         print(h_n, "Each row represents a sentence")
#         x = h_n
#         # print(h_n)
#         # print(c_n)

