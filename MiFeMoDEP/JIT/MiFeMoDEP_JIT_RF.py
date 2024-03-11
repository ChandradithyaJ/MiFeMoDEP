"""
Train the Random Forest Classifier
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time, pickle
import torch
import torch.nn as nn
from gensim.models.doc2vec import Doc2Vec
from transformers import AutoTokenizer, AutoModel
from MiFeMoDEP_JIT_Utils import *

path_to_jit_random = '../DefectorsDataset/defectors/jit_bug_prediction_splits/random'

train_jit_df = pd.read_parquet(f'{path_to_jit_random}/train.parquet.gzip')
train_jit_df = train_jit_df.reset_index(drop=True)

# create file level labels
train_jit_df['target'] = train_jit_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)


train_jit_df['content'] = train_jit_df['content'].apply(preprocess_diff)

train_jit_df['code'] = train_jit_df['content'].apply(preprocess_diff_to_code)


cb_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base')
d2v_model = Doc2Vec.load("./doc2vec_model_train_jit.bin")

# get CodeBERT and PDG Embeddings of each
start_idx = 0
stop_idx = 42
upper_bound = 42
modified_train_source_code_df = pd.DataFrame()
data_list = []
for i in range(0, upper_bound): # for loop to save memory
    while start_idx < upper_bound:  
        rows = [i for i in range(start_idx, max(stop_idx, upper_bound))]    
        temp = train_jit_df.iloc[rows]
        temp['cb_embeds'] = None
        temp['pdg_enc_inp'] = None
        start = time.time()

        temp['cb_embeds'] = temp['content'].apply(lambda x : get_CodeBERT_context_embeddings(cb_tokenizer, cb_model, x))
        temp['pdg_enc_inp'] = temp['content'].apply(lambda x : get_PDG_embeddings(d2v_model, x))
        
        print(start_idx, stop_idx, time.time()-start)
        data_list.append(temp)
        start_idx = stop_idx
        stop_idx += 30

if len(data_list) > 0:
    modified_train_jit_df = pd.concat(data_list)
else:
    modified_train_jit_df = data_list[0]


loaded_dict = torch.load('./MiFeMoDEP_JIT_PDG_Enc.pt')
enc_model.load_state_dict(loaded_dict)


train_dataset = DefectorsTorchDataset(modified_train_jit_df)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(modified_train_jit_df), shuffle=True)

# MiFeMoDEP
y_gt = None
for cb_embeds, pdg_enc_inps, y in train_data_loader:
    pdg_embeds = encode_PDG(torch.unsqueeze(pdg_enc_inps, dim=1))
    embeds = torch.concat((cb_embeds, pdg_embeds), dim=1)
    y_gt = y
    
embeds = embeds.resize(42, 10001*768)
embeds = embeds.detach().numpy()
y_gt = y_gt.detach().numpy()

clf = RandomForestClassifier(n_estimators=100, random_state=101)
clf.fit(embeds, y_gt)
pickle.dump(clf, open('./MiFeMoDEP_JIT_RF.pkl', 'wb'))

