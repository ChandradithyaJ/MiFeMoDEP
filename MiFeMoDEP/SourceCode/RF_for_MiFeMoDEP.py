from RGCN import RGCN
import torch
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle, os, time, h5py

def train_MiFeMoDEP():

    # load the extracted and encoded graphs, and the pretrained RGCN
    edge_index_list = pickle.load(open('./edge_index.pkl', 'rb'))
    edge_types_list = pickle.load(open('./edge_types_list.pkl', 'rb')) 
    node_features_list = pickle.load(open('./node_features_list.pkl', 'rb'))
    labels_list = pickle.load(open('./labels_list.pkl', 'rb'))

    in_channels = 150
    hidden_channels = 100
    out_channels = 50
    num_relations = 21

    RGCN_model = RGCN(in_channels, hidden_channels, out_channels, num_relations)
    RGCN_model_weights = torch.load('./MiFeMoDEP_SourceCode_CPG_Enc.pt')
    RGCN_model.load_state_dict(RGCN_model_weights)

    # load the CodeBERT embeddings
    with h5py.File("all_cb_embeds_transformed.h5", "r") as f:
        dataset = f["all_cb_embeds_transformed"]
        all_cb_embeds_transformed = dataset[:]
        del dataset
    
    # match the corresponding CodeBERT and CPG embeddings
    train_df = pd.read_csv('./random_1200_balanced.csv').drop(['Unnamed: 0'], axis=1)

    X_train = np.empty((0, 2*128))
    y_train = []

    root_dir_graphs = "./Preprocessing_joern/preprocess_graphs_1200/"
    for root, dirs, files in os.walk(root_dir_graphs):
        for dir_str in dirs:
            req_idx = int(dir_str)
            cb_embeds = all_cb_embeds_transformed[req_idx].reshape(1, 128)
            cpg_embeds = RGCN_model(node_features_list[req_idx],edge_index_list[req_idx].to(torch.int64),edge_types_list[req_idx].to(torch.int64)).detach().numpy().reshape(1, 128)
            embeds = np.vstack((cb_embeds, cpg_embeds))
            X_train = np.append(X_train, embeds.reshape(1, 2*128), axis=0)
            y_train.append(labels_list[req_idx])
        break

    y_train = np.array(y_train)
    pickle.dump(X_train, open('X_train.pkl', 'wb'))

    clf = RandomForestClassifier(n_estimators=100, random_state=101)
    start = time.time()
    clf.fit(X_train, y_train)
    print(f"RF Classifier fit in {time.time()-start}s")
    pickle.dump(clf, open('MiFeMoDEP_SourceCode_RF.pkl', 'wb'))

def test_MiFeMoDEP():

    # load the extracted and encoded graphs, and the pretrained RGCN
    edge_index_list = pickle.load(open('./test_edge_index.pkl', 'rb'))
    edge_types_list = pickle.load(open('./test_edge_types_list.pkl', 'rb')) 
    node_features_list = pickle.load(open('./test_node_features_list.pkl', 'rb'))
    labels_list = pickle.load(open('./test_labels_list.pkl', 'rb'))

    in_channels = 150
    hidden_channels = 100
    out_channels = 50
    num_relations = 21

    RGCN_model = RGCN(in_channels, hidden_channels, out_channels, num_relations)
    RGCN_model_weights = torch.load('./MiFeMoDEP_SourceCode_CPG_Enc.pt')
    RGCN_model.load_state_dict(RGCN_model_weights)

    # load the CodeBERT embeddings
    with h5py.File("all_test_cb_embeds_transformed.h5", "r") as f:
        dataset = f["all_cb_embeds_transformed"]
        all_test_cb_embeds_transformed = dataset[:]
        del dataset
    
    # match the corresponding CodeBERT and CPG embeddings
    test_df = pd.read_csv('./balanced_500_test_source_code.csv').drop(['Unnamed: 0'], axis=1)

    X_test = np.empty((0, 2*128))
    y_test = []

    root_dir_graphs = "./Preprocessing_joern/preprocess_graphs_500/"
    for root, dirs, files in os.walk(root_dir_graphs):
        for dir_str in dirs:
            req_idx = int(dir_str)
            cb_embeds = all_test_cb_embeds_transformed[req_idx].reshape(1, 128)
            cpg_embeds = RGCN_model(node_features_list[req_idx],edge_index_list[req_idx].to(torch.int64),edge_types_list[req_idx].to(torch.int64)).detach().numpy().reshape(1, 128)
            embeds = np.vstack((cb_embeds, cpg_embeds))
            X_test = np.append(X_test, embeds.reshape(1, 2*128), axis=0)
            y_test.append(labels_list[req_idx])
        break
    y_test = np.array(y_test)
    pickle.dump(X_test, open('X_test.pkl', 'wb'))

    clf = pickle.load(open('./MiFeMoDEP_SourceCode_RF.pkl', 'rb'))
    y_pred = clf.predict(X_test)

    test_df['y_pred'] = pd.Series(y_pred)
    test_df.to_csv('./balanced_500_test_source_code.csv')

if __name__ == "__main__":
    # train_MiFeMoDEP()
    test_MiFeMoDEP()