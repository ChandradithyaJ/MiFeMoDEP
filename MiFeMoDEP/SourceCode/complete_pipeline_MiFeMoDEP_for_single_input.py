from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from networkx.drawing.nx_pydot import read_dot
from get_CodeBERT_embeddings import get_CodeBERT_context_embeddings
from train_PCA_for_MiFeMoDEP import PCA_single_input
from RGCN import RGCN
from rgcn_training import get_graph_data
from LIME_for_MiFeMoDEP import preprocess_feature_from_explainer, add_agg_scr_to_list
import pickle
import dill
import subprocess
import torch, os
import numpy as np, pandas as pd

filepath = './single.py'
code_string = open(filepath, 'r').read()

# get the CodeBERT embeddings and perform PCA on them
cb_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base')
max_len = 10000
pca = pickle.load(open('./MiFeMoDEP_PCA.pkl', 'rb'))
num_features = 1450000

cb_embeds = get_CodeBERT_context_embeddings(cb_tokenizer, cb_model, code_string, max_len)
cb_embeds = cb_embeds.detach().numpy().reshape(1, -1)
cb_embeds = PCA_single_input(pca, cb_embeds, num_features)

# extract the Code Property Graph using JOERN and encode it using an RGCN
in_channels = 150
hidden_channels = 100
out_channels = 50
num_relations = 21
truncate_size = 2000
RGCN_model = RGCN(in_channels, hidden_channels, out_channels, num_relations)
RGCN_model_weights = torch.load('./MiFeMoDEP_SourceCode_CPG_Enc.pt')
RGCN_model.load_state_dict(RGCN_model_weights)

graph_dir_path = ""
def extract_graph_from_file(filename, output_name):
    result = subprocess.run(["joern-parse", filename], capture_output=True)
    output = result.stdout.decode()
    if result.returncode == 0:
        subprocess.run(["joern-export", "--repr=all", "--out", graph_dir_path+output_name]) 
        

if os.path.isdir("./test_graph"):
    if os.path.exists("./test_graph/export.dot"):
        os.remove("./test_graph/export.dot")
    os.rmdir("./test_graph")

extract_graph_from_file(filepath,"test_graph")

graph_path = graph_dir_path+"test_graph/export.dot"

graph = read_dot(graph_path)
node_features, edge_index, edge_types = get_graph_data(graph,truncate_size,padding=True) # includes Doc2Vec for node embedding
cpg_embeds = RGCN_model(node_features, edge_index.to(torch.int64), edge_types.to(torch.int64)).detach().numpy().reshape(1, 128)

# combine the embeddings and pass it to an RF Classifier
embeds = np.vstack((cb_embeds, cpg_embeds)).reshape(1, 128*2) # changed to add two 128 , 128 embedds
clf = pickle.load(open('./MiFeMoDEP_SourceCode_RF.pkl', 'rb'))
y_pred = clf.predict(embeds)
if y_pred == 1:
    # use LIME explainer
    explainer = dill.load(open('LIME_for_MiFeMoDEP.pkl', 'rb'))
    exp = explainer.explain_instance(
        embeds.reshape(256),
        clf.predict_proba,
        num_features=256,
        top_labels=1,
        num_samples=5000
    )

    top_k_tokens = np.arange(10,201,10)
    agg_methods = ['avg','median','sum']
    max_str_len_list = 100
    max_tokens = 100
    line_score_df_col_name = ['total_tokens', 'line_level_label', 'line_num'] + ['token'+str(i) for i in range(1,max_str_len_list+1)] + [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]

    line_score_df = pd.DataFrame(columns=line_score_df_col_name)  
    line_score_df = line_score_df.set_index('line_num')

    sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(exp)

    code_lines = code_string.splitlines()
    for line_num, line in enumerate(code_lines):
        if type(line) == float: # nan
            line = ""
            
        line_stuff = []
        line_score_list = np.zeros(max_tokens)
        token_list = line.split()[:max_tokens]
        line_stuff.append(line)
        line_stuff.append(len(token_list))
        
        for tok_idx, tok in enumerate(token_list):
            score = sorted_feature_score_dict.get(tok, 0)
            line_score_list[tok_idx] = score
            
        line_stuff = line_stuff + list(line_score_list)
        
        for k in top_k_tokens:
            top_tokens = tokens_list[:k-1]
            top_k_scr_list = []
            
            if len(token_list) < 1:
                top_k_scr_list.append(0)
            else:
                for tok in token_list:
                    score = 0
                    if tok in top_tokens:
                        score = sorted_feature_score_dict.get(tok,0)
                    top_k_scr_list.append(score)

            add_agg_scr_to_list(line_stuff, top_k_scr_list)

        add_agg_scr_to_list(line_stuff, list(line_score_list[:len(token_list)]))
        line_score_df.loc[line_num] = line_stuff
    line_score_df.to_csv('./single_df.csv')
    scr_df = line_score_df['median-all-tokens'].values.tolist()

    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(scr_df).reshape(-1, 1))
    line_df = pd.DataFrame()
    line_df['scr'] = [float(val.item()) for val in line_score]
    line_df = line_df.sort_values(by='scr',ascending=True)

    buggy_order = [idx+1 for idx,row in line_df.iterrows()]
    print('The possible buggy lines in order of most to least probable:\n',buggy_order)

else:
    print('No buggy lines')