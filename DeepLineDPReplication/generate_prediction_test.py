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

test_df = pd.read_csv('preprocessed_test_df.csv')
test_df['lines'] = test_df['lines'].apply(lambda x: string_to_list(x))

filepath = 'prediction_DeepLineDP.csv'
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


def test_model():

    start = time.time()
    test_code3d, test_label = get_code3d_and_label(test_df, 'test')
    print(f"code_3d done in {time.time()-start}s")
    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(test_label), y = test_label)
    weight_dict = {}
    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)

    word2vec = Word2Vec.load('./word2vec/w2v50dim.bin')
    print('load Word2Vec finished')

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)
    vocab_size = len(word2vec.wv.key_to_index) + 1  # Use key_to_index
    # for unknown tokens

    x_test_vec = get_x_vec(test_code3d, word2vec)

    max_sent_len = min(max([len(sent) for sent in (x_test_vec)]), max_test_LOC)

    test_dl = get_dataloader(x_test_vec,test_label,batch_size,max_sent_len)

    loaded_dict = torch.load('./DeepLineDPReplication.pth')
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
        dropout=dropout
    )
    model.load_state_dict(loaded_dict)

    model.eval()

    batch_start = 0
    for inputs, labels in test_dl:
        start = time.time()
        outputs, word_att_weights, sent_att_weights, sents = model(inputs)
        outputs = outputs.detach().numpy()
        word_att_weights = word_att_weights.detach().numpy()
        sent_att_weights = sent_att_weights.detach().numpy()
        labels = labels.detach().numpy()

        is_comment = False
        columns = ['commit', 'repo', 'filepath', 'is_comment', 'line', 'is_buggy', 'attention_score', 'file_target', 'ypred_file']
        final_test_pred = pd.DataFrame(columns=columns)

        row_dict = []
        for i in range(batch_start, batch_start+batch_size):
            lines = test_df['content'].iloc[i].split('\n')
            j = 0  # line number in the file

            print(f'doc num:{i}')

            is_comment = False
            pred_i = i - batch_start
            for line in lines:
                j += 1
                row_dict = []
                row_dict.append(test_df['commit'][i])
                row_dict.append(test_df['repo'][i])
                row_dict.append(test_df['filepath'][i])
                if line.startswith('#'):
                    row_dict.append(True)
                    row_dict.append(line)
                    row_dict.append(False)
                elif line.startswith("'''"):
                    is_comment = True
                    row_dict.append(True)
                    row_dict.append(line)
                    row_dict.append(False)
                elif is_comment:
                    row_dict.append(True)
                    row_dict.append(line)
                    row_dict.append(False)
                elif line.endswith("'''") and is_comment:
                    is_comment = False
                    row_dict.append(True)
                    row_dict.append(line)
                    row_dict.append(False)
                elif line == '':
                    continue
                else:
                    row_dict.append(False)
                    row_dict.append(line)
                    row_dict.append(False if j not in test_df['lines'][i] else True)
                # tdf[attention_score]. Each word score must be updated from word attention weights in the test model output
                if j < len(word_att_weights):
                    attention_score = 0.0
                    for k in range(len(word_att_weights[j])):
                        attention_score += word_att_weights[j][k].sum()
                    # row_dict.append(np.array([attention_score]))
                    print('attention', type(attention_score))
                    row_dict.append(attention_score)
                    break
                else:
                    row_dict.append(None)  # or some other default value
                
                # fill tdf[file_target] with the y_gt from test_df
                if pred_i < len(labels):
                    row_dict.append(labels[pred_i])
                else:
                    row_dict.append(None) 
                # fill tdf[ypred_file] with the y_pred from test_df
                if pred_i < len(outputs):
                    row_dict.append(outputs[pred_i])
                else:
                    row_dict.append(None) 
                final_test_pred.loc[len(final_test_pred.index)] = row_dict

        del outputs, word_att_weights, sent_att_weights, sents

    return final_test_pred
final_test_pred = test_model()

final_test_pred['ypred_file'] = final_test_pred['ypred_file'].apply(lambda x : 1 if x[0] >= 0.375 else 0)

final_test_pred.to_csv(filepath) # saving the daaframe as "prediction_DeepLineDP.csv"  
