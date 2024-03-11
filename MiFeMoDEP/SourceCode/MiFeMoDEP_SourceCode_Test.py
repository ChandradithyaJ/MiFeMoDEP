import pandas as pd
import time, math, pickle
import sklearn.metrics as metrics
from gensim.models.doc2vec import Doc2Vec
from transformers import AutoTokenizer, AutoModel
from MiFeMoDEP_SourceCode_Utils import *

path_to_source_code_random = '../DefectorsDataset/defectors/line_bug_prediction_splits/random'

test_source_code_df = pd.read_parquet(f'{path_to_source_code_random}/test.parquet.gzip')
test_source_code_df = test_source_code_df.reset_index(drop=True)

test_source_code_df['target'] = test_source_code_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

test_source_code_df['content'] = test_source_code_df['content'].apply(lambda x : '' if x is None else x.decode("latin-1"))


cb_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base')
d2v_model = Doc2Vec.load("./doc2vec_model_train_source_code.bin")


# get CodeBERT and PDG Embeddings of each
# 27 - 45 mins for 42 docs (cache?)
start_idx = 300
stop_idx = 310
upper_bound = 346
modified_train_source_code_df = pd.DataFrame()
data_list = []
for i in range(0, upper_bound): # for loop to save memory
    while start_idx < upper_bound:
        rows = [i for i in range(start_idx, max(stop_idx, upper_bound)) if i != 443]    
        temp = test_source_code_df.iloc[rows]
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
    modified_test_source_code_df = pd.concat(data_list)
else:
    modified_test_source_code_df = data_list[0]


# In[26]:


loaded_dict = torch.load('./MiFeMoDEP_SourceCode_PDG_Enc.pt')
enc_model.load_state_dict(loaded_dict)


test_dataset = DefectorsTorchDataset(modified_test_source_code_df)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(modified_test_source_code_df), shuffle=True)


# MiFeMoDEP
y_gt = None
for cb_embeds, pdg_enc_inps, y in test_data_loader:
    pdg_embeds = encode_PDG(torch.unsqueeze(pdg_enc_inps, dim=1))
    embeds = torch.concat((cb_embeds, pdg_embeds), dim=1)
    y_gt = y
    
embeds = embeds.resize(17, 10001*768)
embeds = embeds.detach().numpy()
y_gt = y_gt.detach().numpy()


with open("MiFeMoDEP_SourceCode_RF.pkl", "rb") as file:
    clf = pickle.load(file)

# Make predictions using the loaded model
y_pred = clf.predict(embeds)
y_pred_probas = clf.predict_proba(embeds)

# precision, Recall, F1-score, Confusion matrix, False Alarm Rate, Distance-to-Heaven, AUC
prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_gt,y_pred,average='binary') # at threshold = 0.5
tn, fp, fn, tp = metrics.confusion_matrix(y_gt, y_pred, labels=[0, 1]).ravel()
FAR = fp/(fp+tn)
dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0)
AUC = metrics.roc_auc_score(y_gt, y_pred_probas)

print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-score: {f1}")
print(f"False Alarm Rate: {FAR}")
print(f"Distance to Heaven: {dist_heaven}")
print(f"AUC: {AUC}")

