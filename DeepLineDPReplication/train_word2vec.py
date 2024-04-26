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

train_df = pd.read_csv('./preprocessed_train_df.csv')
test_df = pd.read_csv('./preprocessed_test_df.csv')


def train_word2vec_model(embedding_dim = 50):

    w2v_path = './word2vec'

    save_path = w2v_path+'/'+'w2v'+str(embedding_dim)+'dim.bin'

    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    train_code_3d, _ = get_code3d_and_label(train_df)code2d
    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))
    return word2vec

word2vec = train_word2vec_model()
