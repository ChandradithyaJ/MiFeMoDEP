import pandas as pd
import time
import networkx as nx
from networkx.drawing.nx_pydot import read_dot,write_dot

import os 

path_to = "./"
graph_root_dir = "Preprocessing_joern/preprocess_graphs_1200/"
file_path_list = []
node_num_list = []
dictionary_list = []

# converting graph information in .dot files to a .csv for faster training of RGCN
def write_to_df(path):
    G = read_dot(path+"/export.dot")
    nodes = G.nodes(data=True)
    for node in nodes:
        num,dic = node
        file_path_list.append(path)
        node_num_list.append(num)
        dictionary_list.append(str(dic))
 
for root,dirs,files in os.walk(path_to+graph_root_dir):
    for file in files:
        path = os.path.join(root,file)
        start = time.time()
        write_to_df(os.path.dirname(path))
        end = time.time()
        print("time: ",end-start)

df = pd.DataFrame({'file_path':file_path_list,'node_num':node_num_list,'dictionary':dictionary_list})
df.to_csv('nodes_all_graphs.csv')
