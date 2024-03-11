import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import compute_class_weight
from DeepLineDP_Utils import *
import sklearn.metrics as metrics
import math

path_to_line_random = '../DefectorsDataset/defectors/line_bug_prediction_splits/random'

train_df = pd.read_parquet(f'{path_to_line_random}/train.parquet.gzip')
train_df = train_df.reset_index(drop=True)

test_df = pd.read_parquet(f'{path_to_line_random}/test.parquet.gzip')
test_df = test_df.reset_index(drop=True)

train_df['target'] = train_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
test_df['target'] = test_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

train_df_1 = train_df[train_df['target'] == 1].sample(75, random_state=42)
train_df_0 = train_df[train_df['target'] == 0].sample(75, random_state=42)

# Combine the DataFrames
train_df = pd.concat([train_df_1, train_df_0], ignore_index=True)


train_df['content'] = train_df['content'].apply(preprocess_code)
train_df.to_csv('./preprocessed_train_df.csv')


# Token Embedding Layer
train_df['content'] = train_df['content'].apply(prepare_code2d)
train_df.to_csv('./token2d_train_df.csv')

test_df = test_df.head(100)
test_df['content'] = test_df['content'].apply(preprocess_code)
test_df['content'] = test_df['content'].apply(prepare_code2d)


word2vec = train_word2vec_model(df=train_df)
  
# Model Training

torch.manual_seed(0)

# model setting
batch_size = 32 #args.batch_size
num_epochs = 10 #args.num_epochs
max_grad_norm = 5
embed_dim = 50 #args.embed_dim
word_gru_hidden_dim = 64 #args.word_gru_hidden_dim
sent_gru_hidden_dim = 64 #args.sent_gru_hidden_dim
word_gru_num_layers = 1 #args.word_gru_num_layers
sent_gru_num_layers = 1 #args.sent_gru_num_layers
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = 0.2 #args.dropout
lr = 0.001 #args.lr
max_train_LOC = 900

def train_model():


    train_code3d, train_label = get_code3d_and_label(train_df)

    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_label), y = train_label)

    weight_dict['defect'] = np.max(sample_weights)
    weight_dict['clean'] = np.min(sample_weights)

    word2vec = Word2Vec.load('./word2vec/w2v50dim.bin')
    print('load Word2Vec finished')

    word2vec_weights = get_w2v_weight_for_deep_learning_models(word2vec, embed_dim)

    vocab_size = len(word2vec.wv.key_to_index) + 1  # Use key_to_index
    # for unknown tokens

    x_train_vec = get_x_vec(train_code3d, word2vec)

    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)


    train_dl = get_dataloader(x_train_vec,train_label,batch_size,max_sent_len)

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

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        start = time.time()
        train_losses = []

        model.train()

        for inputs, labels in train_dl:

            output, _, __, ___ = model(inputs)

            weight_tensor = get_loss_weight(labels)

            criterion.weight = weight_tensor

            loss = criterion(output, labels.reshape(batch_size,1))

            train_losses.append(loss.item())
            
            

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
        print(f'Epoch {epoch}: {time.time()-start}')

    torch.save(model.state_dict(), './DeepLineDPReplication.pth')

train_model()

batch_size = len(test_df)
torch.manual_seed(0)
embed_dim = 50 #args.embed_dim
word_gru_hidden_dim = 64 #args.word_gru_hidden_dim
sent_gru_hidden_dim = 64 #args.sent_gru_hidden_dim
word_gru_num_layers = 1 #args.word_gru_num_layers
sent_gru_num_layers = 1 #args.sent_gru_num_layers
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = 0.2 #args.dropout
lr = 0.001 #args.lr
max_test_LOC = 900

def test_model():


    test_code3d, test_label = get_code3d_and_label(test_df)

    sample_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(test_label), y = test_label)

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

    start = time.time()

    model.eval()

    for inputs, labels in test_dl:
        outputs, _, __, ___ = model(inputs)
        return outputs, labels

y_pred, y_gt = test_model()
y_pred = y_pred.detach().numpy()
y_gt = y_gt.detach().numpy()

y_probs = np.array([prob[0] for prob in y_pred])
y_pred = np.where(y_probs >= 0.45, 1, 0)

# precision, Recall, F1-score, Confusion matrix, False Alarm Rate, Distance-to-Heaven, AUC
prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_gt,y_pred,average='binary') # at threshold = 0.5
tn, fp, fn, tp = metrics.confusion_matrix(y_gt, y_pred, labels=[0, 1]).ravel()
FAR = fp/(fp+tn)
dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0)
AUC = metrics.roc_auc_score(y_gt, y_probs)

print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-score: {f1}")
print(f"False Alarm Rate: {FAR}")
print(f"Distance to Heaven: {dist_heaven}")
print(f"AUC: {AUC}")

