import pandas as pd
import time, math, pickle
from MiFeMoDEP_JIT_Utils import *
import torch
from gensim.models.doc2vec import Doc2Vec
from transformers import AutoTokenizer, AutoModel
import sklearn.metrics as metrics

path_to_jit_random = './../Datasets/defectors/jit_bug_prediction_splits/random'

test_jit_df = pd.read_parquet(f'{path_to_jit_random}/test.parquet.gzip')
test_jit_df = test_jit_df.reset_index(drop=True)


test_jit_df['target'] = test_jit_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
test_jit_df['content'] = test_jit_df['content'].apply(preprocess_diff)

test_jit_df['code'] = test_jit_df['content'].apply(preprocess_diff_to_code)

cb_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base')
d2v_model = Doc2Vec.load("./doc2vec_model_train_jit.bin")

# get CodeBERT and PDG Embeddings of each
# 3.5 minutes for 5 embeddings ==> 100 embeddings = 70 mins
start_idx = 0
stop_idx = 150
siz = stop_idx
upper_bound = 150
modified_train_source_code_df = pd.DataFrame()
data_list = []
for i in range(0, upper_bound): # for loop to save memory
    while start_idx < upper_bound:  
        rows = [i for i in range(start_idx, max(stop_idx, upper_bound))]    
        temp = test_jit_df.iloc[rows]
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
    modified_test_jit_df = pd.concat(data_list)
else:
    modified_test_jit_df = data_list[0]

loaded_dict = torch.load('./MiFeMoDEP_JIT_PDG_Enc.pt')
enc_model.load_state_dict(loaded_dict)

test_dataset = DefectorsTorchDataset(modified_test_jit_df)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(modified_test_jit_df), shuffle=True)

# MiFeMoDEP
y_gt = None
for cb_embeds, pdg_enc_inps, y in test_data_loader:
    pdg_embeds = encode_PDG(torch.unsqueeze(pdg_enc_inps, dim=1))
    embeds = torch.concat((cb_embeds, pdg_embeds), dim=1)
    y_gt = y
    
embeds = embeds.resize(siz, 10001*768)
embeds = embeds.detach().numpy()
y_gt = y_gt.detach().numpy()

clf = pickle.load(open('./MiFeMoDEP_JIT_RF.pkl', 'rb'))
y_pred = clf.predict(embeds)
y_probas = clf.predict_proba(embeds)

# precision, Recall, F1-score, Confusion matrix, False Alarm Rate, Distance-to-Heaven, AUC
prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true=y_gt,y_pred=y_pred,average='binary') # at threshold = 0.5
tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_gt,y_pred= y_pred, labels=[0, 1]).ravel()
FAR = fp/(fp+tn)
dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0)
AUC = metrics.roc_auc_score(y_true=y_gt,y_score= y_probas[:,1])

print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-score: {f1}")
print(f"False Alarm Rate: {FAR}")
print(f"Distance to Heaven: {dist_heaven}")
print(f"AUC: {AUC}")