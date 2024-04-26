import subprocess
import pandas as pd

# #loading train dataframe
path_to_line_random = '.'
code_files_root = "./Preprocessing_joern_test/preprocess_files_500/"
graph_files_root = "./Preprocessing_joern_test/preprocess_graphs_500/"

# train_df = pd.read_parquet(f'{path_to_line_random}/train.parquet.gzip')
# train_df = train_df.reset_index(drop=True)

# train_df['target'] = train_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

# train_df_1 = train_df[train_df['target'] == 1].sample(600, random_state=42)
# train_df_0 = train_df[train_df['target'] == 0].sample(600, random_state=42)
# print(len(train_df_0))
# print(len(train_df_1))

# # Combine the DataFrames
# train_df = pd.concat([train_df_1, train_df_0], ignore_index=True)
# train_df['target'] = train_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
# content_list = train_df['content']
# train_df.drop('content',axis=1,inplace=True)
# train_df.to_csv("random_1200_balanced.csv")
# train_df['content'] = content_list.apply(lambda x : '' if x is None else x.decode("latin-1"))

train_df = pd.read_csv("balanced_500_test_source_code.csv")
train_df['content'] = train_df["content"].apply(lambda x : '' if x is None else x)
train_df["content"] = train_df["content"].apply(lambda x: '' if type(x)==float else x)

print(train_df["content"][327])

def extract_graph_from_file(filename, output_name):
    global graph_files_root
    result = subprocess.run(["joern-parse", filename], capture_output=True)
    output = result.stdout.decode()
    if result.returncode == 0:
        subprocess.run(["joern-export", "--repr=all", "--out", graph_files_root+output_name]) 

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
