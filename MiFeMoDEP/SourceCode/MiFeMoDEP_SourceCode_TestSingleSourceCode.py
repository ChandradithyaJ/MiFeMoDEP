import pickle
import torch
from MiFeMoDEP_SourceCode_Utils import *
from gensim.models.doc2vec import Doc2Vec
from transformers import AutoTokenizer, AutoModel

cb_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base')
d2v_model = Doc2Vec.load("./doc2vec_model_train_source_code.bin")
loaded_dict = torch.load('./MiFeMoDEP_SourceCode_PDG_Enc.pt')
enc_model.load_state_dict(loaded_dict)
with open('./MiFeMoDEP_SourceCode_RF.pkl', 'rb') as f:
    rf = pickle.load(f)


##########################
path_to_code = ""
##########################

with open(path_to_code, "r") as file:
    code_string = file.read()

code_string = preprocess_diff(code_string)

cb_embeds = get_CodeBERT_context_embeddings(cb_tokenizer, cb_model, code_string)
pdg_enc_inp = get_PDG_embeddings(d2v_model, code_string)
pdg_embeds = encode_PDG(torch.unsqueeze(pdg_enc_inp, dim=1))

embeds = torch.concat((cb_embeds, pdg_embeds), dim=1)
embeds = embeds.detach().numpy()

pred = rf.predict(embeds)

if pred[0] == 0:
    print('Not a bug inducing file')
else:
    print('Bug-inducing file')

