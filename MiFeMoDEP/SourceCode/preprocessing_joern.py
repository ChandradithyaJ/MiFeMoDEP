import subprocess
import pandas as pd

# #loading train dataframe
path_to_line_random = '.'
code_files_root = "./Preprocessing_joern_test/preprocess_files_500/"
graph_files_root = "./Preprocessing_joern_test/preprocess_graphs_500/"

train_df = pd.read_csv("balanced_500_test_source_code.csv")
train_df['content'] = train_df["content"].apply(lambda x : '' if x is None else x)
train_df["content"] = train_df["content"].apply(lambda x: '' if type(x)==float else x)

print(train_df["content"][327])

# extracting CPG graphs using joern
def extract_graph_from_file(filename, output_name):
    global graph_files_root
    result = subprocess.run(["joern-parse", filename], capture_output=True)
    output = result.stdout.decode()
    if result.returncode == 0:
        subprocess.run(["joern-export", "--repr=all", "--out", graph_files_root+output_name]) 

# defectors dataset has code content in dataframe so converting it to a python file for joern to parse
def write_to_file(row,index):
    filename = str(index)
    content = row['content']
    with open(code_files_root+filename+".py", "w") as f:
        f.write(content)
    f.close()

for index, row in train_df.iterrows():
    write_to_file(row,index)

for index, row in train_df.iterrows():
    name = str(index)
    extract_graph_from_file(code_files_root+name+".py", name)
