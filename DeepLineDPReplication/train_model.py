import os, re, time, pickle, csv
import pandas as pd
import numpy as np
import more_itertools
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from sklearn.utils import compute_class_weight
from DeepLineDP_model import *
from my_utils import *

train_df = pd.read_csv('preprocessed_train_df.csv')
train_df['lines'] = train_df['lines'].apply(lambda x: string_to_list(x))

torch.manual_seed(0)

# model setting
batch_size = 32 #args.batch_size
num_epochs = 10 #args.num_epochscode2d
save_every_epochs = 1
exp_name = ''#args.exp_name
max_grad_norm = 5
embed_dim = 50 #args.embed_dim
word_gru_hidden_dim = 64 #args.word_gru_hidden_dim
sent_gru_hidden_dim = 64 #args.sent_gru_hidden_dim
word_gru_num_layers = 2 #args.word_gru_num_layers
sent_gru_num_layers = 2 #args.sent_gru_num_layers
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = 0.1 #args.dropout
lr = 0.001 #args.lr

max_train_LOC = 900

weight_dict = {}

train_df_1 = train_df[train_df['target'] == 1].sample(5000, random_state=42)
train_df_0 = train_df[train_df['target'] == 0].sample(5000, random_state=42)

# Combine the DataFrames
train_df = pd.concat([train_df_1, train_df_0], ignore_index=True)


def train_model():

    start = time.time()
    train_code3d, train_label = get_code3d_and_label(train_df, 'train')
    print(f"code_3d done in {time.time()-start}s") # 

    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_label), y = train_label)

    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)

    word2vec = Word2Vec.load('./word2vec/w2v50dim.bin') # loading trained word2vec 
    print('load Word2Vec finished')  

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)
    vocab_size = len(word2vec.wv.key_to_index) + 1
    x_train_vec = get_x_vec(train_code3d, word2vec)
    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)


    train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len) # dataloader in my_utils

    model = HierarchicalAttentionNetwork(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        word_gru_hidden_dim=word_gru_hidden_dim,
        sent_gru_hidden_dim=sent_gru_hidden_dim,
        word_gru_num_layers=word_gru_num_layers,
        sent_gru_num_layers=sent_gru_num_layers,
        word_att_dim=word_att_dim,
        sent_att_dim=sent_att_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout)

    model.sent_attention.word_attention.freeze_embeddings(False)
    
    print(f"num_batches: {int(train_df.shape[0]/batch_size)}")

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        start = time.time()
        train_losses = []

        model.train()
        batch_num = 0
        batch_start = time.time()
        for inputs, labels in train_dl:

            output,word_att_weights,sent_att_weights,sents = model(inputs)

            weight_tensor = get_loss_weight(labels)

            criterion.weight = weight_tensor

            loss = criterion(output, labels.reshape(batch_size,1))

            train_losses.append(loss.item())  

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            if batch_num%50 == 0:
                print(batch_num, f'{time.time()-batch_start}')
                batch_start = time.time()

            batch_num += 1

        print(f'Epoch {epoch}: {time.time()-start}')

        torch.save(model.state_dict(), './DeepLineDPReplication.pth')


train_model()