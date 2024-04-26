import time
import os, pickle
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
import numpy as np
from torch import nn
from networkx.drawing.nx_pydot import read_dot

from gensim.models.doc2vec import Doc2Vec

from torch_geometric.data import Dataset,Data
from torch_geometric.nn import FastRGCNConv, RGCNConv,SAGPooling
from torch_geometric.utils import k_hop_subgraph

# Define device for computations (CPU or GPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("device: ",device)

path_to_doc2vec = "./"
doc2vec_model = Doc2Vec.load(path_to_doc2vec+"doc2vec_model_nodes_from_graphs.bin")

e_types = ['RECEIVER', 'CDG', 'CFG', 'CONDITION', 'BINDS', 'REACHING_DEF', 'PARAMETER_LINK', 'IS_CALL_FOR_IMPORT', 'POST_DOMINATE', 'AST', 'CALL', 'REF', 'CONTAINS', 'INHERITS_FROM', 'TAGGED_BY', 'SOURCE_FILE', 'ARGUMENT', 'CAPTURE', 'DOMINATE', 'EVAL_TYPE']
def get_edge_num(edge_type):
    for i in range(20):
        if(edge_type == e_types[i]):
            return i
    return 20

def get_graph_data(graph, truncate_size, padding=False):
  edges = graph.edges(data=True)
  nodes = graph.nodes(data=True)
  mapping = {}
  dictionary_list = []
  i=0
  for node in nodes:
      if i >= truncate_size:
        break
      num,dic = node
      # print(num, end=',')
      mapping[num] = i
      dictionary_list.append(doc2vec_model.infer_vector([str(dic)])) 
      i += 1
  nodes_features = torch.Tensor(np.array(dictionary_list))
  #   changed_nodes = [x for x in range(i)]
  #   changed_nodes = torch.Tensor(np.array(changed_nodes)).int()
  # print(changed_nodes)
  # edge_idx,edge_type = get_edges(edges)
  # Computing edges
  edge_index_1 = []
  edge_index_2 = []
  edge_type = []
  for edge in edges:
    t0,t1,t2 = edge
    # print(t0, t1, sep=',', end = '|')
    if t0 not in mapping or t1 not in mapping:
      continue 
    edge_index_1.append(int(mapping[t0]))
    edge_index_2.append(int(mapping[t1]))
    edge_type.append(get_edge_num(t2['label']))
  edge_index = [edge_index_1,edge_index_2]
  edge_idx,edge_type = torch.Tensor(np.array(edge_index)).to(torch.int64),torch.Tensor(np.array(edge_type)).to(torch.int64)
  # print(edge_idx)
  # nodes_features = extract_features(nodes)
  while(padding and i < truncate_size):
    # changed_nodes = torch.cat((changed_nodes,torch.Tensor([i]).int()),0)
    nodes_features = torch.cat((nodes_features,torch.Tensor([0 for x in range(150)]).view(1,-1)),0)
    i += 1

  return nodes_features, edge_idx, edge_type

if __name__ == "__main__":
    
    truncate_size = 2000

    class RGCNDataset(Dataset):
        def __init__(self, edge_index_list=[], edge_types_list=[], node_features_list=[], labels_list=[]):
            super(RGCNDataset, self).__init__()
            self.edge_index_list = edge_index_list
            self.node_features_list = node_features_list
            self.edge_types_list = edge_types_list
            self.labels_list = labels_list

        def __len__(self):
            return len(self.edge_index_list)  # Length based on number of graphs

        def __getitem__(self, idx):
            edge_index = self.edge_index_list[idx]
            edge_types = self.edge_types_list[idx]
            node_features = self.node_features_list[idx]
            labels = self.labels_list[idx]
            return edge_index, edge_types, node_features, labels

    in_channels = 150
    hidden_channels = 100
    out_channels = 50
    num_relations = 21
    truncate_size = 2000

    edge_index_list = [0]*500 
    edge_types_list = [0]*500 
    node_features_list = [0]*500
    labels_list = [0]*500

    test_df = pd.read_csv("../Documents/cs21b059/MiFeMoDEP/balanced_500_test_source_code.csv")

    def get_node_and_edge_encodings():
        root_dir_graphs = "./preprocess_graphs_500/"
        count = 0
        for root,dirs,files in os.walk(root_dir_graphs):
            for gfile in files:
                start = time.time()
                path = os.path.join(root,gfile)
                name = root.replace(root_dir_graphs,'')
                i = int(name)
                graph = read_dot(path)
                node_features,edge_index,edge_types = get_graph_data(graph,truncate_size,padding=True)
                label = test_df['target'][i]
                edge_index_list[i] = edge_index
                node_features_list[i] = node_features
                edge_types_list[i] = edge_types
                labels_list[i] = label
                end = time.time()
                print("Time taken for one graph: ",end-start,"sec")
                count += 1
            
                with open('test_edge_index.pkl', 'wb') as f:
                    pickle.dump(edge_index_list, f)

                with open('test_edge_types_list.pkl', 'wb') as f:
                    pickle.dump(edge_types_list, f) 

                with open('test_node_features_list.pkl', 'wb') as f:
                    pickle.dump(node_features_list, f) 

                with open('test_labels_list.pkl', 'wb') as f:
                    pickle.dump(labels_list, f)

                print(count)

    def train_RGCN():
        edge_index_list = pickle.load(open('./edge_index.pkl', 'rb'))
        edge_types_list = pickle.load(open('./edge_types_list.pkl', 'rb')) 
        node_features_list = pickle.load(open('./node_features_list.pkl', 'rb'))
        labels_list = pickle.load(open('./labels_list.pkl', 'rb'))

        class rgcn_2000_nodes(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
                super().__init__()
                self.in_channels = in_channels
                self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
                self.sag_pool = SAGPooling(hidden_channels, ratio=0.8)
                self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
                self.lin = nn.Linear(out_channels*1600, 128)

            def forward(self, x, edge_index, edge_type):
                x = self.conv1(x, edge_index, edge_type)
                x, edge_index, edge_type, _, _, _ = self.sag_pool(x, edge_index, edge_type)
                x = self.conv2(x, edge_index, edge_type)
                x = x.view(x.size(0)*x.size(1))
                x = self.lin(x)
                return x

        class NNClassifier(nn.Module):
            def __init__(self):
                super(NNClassifier, self).__init__()

                self.l1 = nn.Linear(128, 64)
                self.l2 = nn.Linear(64, 1)
                self.leakyrelu = nn.LeakyReLU()
                self.sigmoid = nn.Sigmoid()
                self.rgcn_model = rgcn_2000_nodes(in_channels, hidden_channels, out_channels, num_relations)

            def forward(self, node_features, edge_index, edge_types):
                x = self.rgcn_model(node_features, edge_index, edge_types)
                x = self.l1(x)
                x = self.leakyrelu(x)
                x = self.l2(x)
                pred = self.sigmoid(x)
                return pred

        learning_rate = 0.001
        num_epochs = 100

        rgcn_dataset = RGCNDataset(edge_index_list,edge_types_list,node_features_list,labels_list)
        model = NNClassifier()
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            start = time.time()
            for i in range(rgcn_dataset.__len__()):
                pred = model(node_features_list[i],edge_index_list[i].to(torch.int64),edge_types_list[i].to(torch.int64))
                loss = loss_fn(pred, torch.Tensor([labels_list[i]]))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i%100 == 0:
                    print(i)
                    torch.save(model.state_dict(), "./MiFeMoDEP_SourceCode.pt")
                    torch.save(model.rgcn_model.state_dict(), './MiFeMoDEP_SourceCode_PDG_Enc.pt')

            print(f"Epoch {epoch} --> {time.time()-start}")        
            torch.save(model.state_dict(), "./MiFeMoDEP_SourceCode.pt")
            torch.save(model.rgcn_model.state_dict(), './MiFeMoDEP_SourceCode_PDG_Enc.pt')
