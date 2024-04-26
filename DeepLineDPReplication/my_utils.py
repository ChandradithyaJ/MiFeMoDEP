import os, re, time, pickle, csv
import pandas as pd
import numpy as np
import more_itertools
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from sklearn.utils import compute_class_weight


max_seq_len = 50
def prepare_code2d(code_list, df_name):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    # content to list(content)

    print(f"preparing code3d... len: {len(code_list)}")
    prev_idx = -1
    code3d = []
    for i, file in enumerate(code_list):

        code2d = []
        if type(file) == float:
            file = ""

        file = file.splitlines()

        for c in file:
            c = re.sub('\\s+',' ',c)

            c = c.lower()

            token_list = c.strip().split()
            total_tokens = len(token_list)

            token_list = token_list[:max_seq_len]

            if total_tokens < max_seq_len:
                token_list = token_list + ['<pad>']*(max_seq_len-total_tokens)

            code2d.append(token_list)
            
        code3d.append(code2d)

        if i%100 == 0:
            print(f"prepared code2d {i}")
            if os.path.exists(f"./{df_name}_code3d_{prev_idx}.pkl"):
                os.remove(f"./{df_name}_code3d_{prev_idx}.pkl")
            pickle.dump(code3d, open(f"./{df_name}_code3d_{i}.pkl", "wb"))
            prev_idx = i

    if os.path.exists(f"./{df_name}_code3d_{prev_idx}.pkl"):
            os.remove(f"./{df_name}_code3d_{prev_idx}.pkl")
    pickle.dump(code3d, open(f"./{df_name}_code3d_{len(code_list)}.pkl", "wb"))
    return code3d

def get_code3d_and_label(df, df_name):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''
    if os.path.exists(f'./{df_name}_code3d_{df.shape[0]}.pkl'):
        with open(f'./{df_name}_code3d_{df.shape[0]}.pkl', 'rb') as f:
            code_3d = pickle.load(f)
    else:
        code_3d = prepare_code2d(df['content'].tolist(), df_name)
    all_file_label = df['target'].to_numpy().tolist()
    return code_3d, all_file_label



def get_x_vec(code_3d, word2vec):
    x_vec = [[[
        word2vec.wv.key_to_index[token] if token in word2vec.wv.key_to_index else len(word2vec.wv.key_to_index)
        for token in text
    ] for text in texts] for texts in code_3d]
    return x_vec


def pad_code(code_list_3d, max_sent_len, limit_sent_len=True, mode='train'):
    padded = []

    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            # Truncate if line is longer than max_seq_len
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            # edited here ..just trying
            elif len(line) <= max_seq_len:
                new_line = line + [0] * (max_seq_len - len(new_line))
            sent_list.append(new_line)
        
        # Pad the entire file (all sentences) to max_sent_len with zeros
        padded_file = sent_list + [[0] * max_seq_len for _ in range(max_sent_len - len(sent_list))]

        # If in training mode and `limit_sent_len` is True, keep only the first max_sent_len sentences
        if mode == 'train' and limit_sent_len:
            padded_file = padded_file[:max_sent_len]
        
        padded.append(padded_file)
        # print(padded_file)
    return padded


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.vectors)
    # add zero vector for unknown tokens
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1,embed_dim)))
    return word2vec_weights
    
def get_dataloader(code_vec, label_list, batch_size, max_sent_len):
    y_tensor = torch.FloatTensor([label for label in label_list])
    code_vec_pad = pad_code(code_vec,  max_sent_len)
    # Ensure padding happens to code2d
    tensor_dataset = TensorDataset(torch.tensor(code_vec_pad), y_tensor)
    dl = DataLoader(tensor_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return dl


def string_to_list(lines_str):
    numbers = re.findall("\d+", lines_str)
    num_list = [int(num) for num in numbers]
    return num_list

def get_loss_weight(labels):
    '''
        input
            labels: a PyTorch tensor that contains labels
        output
            weight_tensor: a PyTorch tensor that contains weight of defect/clean class
    '''
    label_list = labels.cpu().numpy().squeeze().tolist()
    weight_list = []

    for lab in label_list:
        if lab == 0:
            weight_list.append(weight_dict['clean'])
        else:
            weight_list.append(weight_dict['defect'])
    weight_tensor = torch.tensor(weight_list).reshape(-1,1)
    return weight_tensor



def top_k_tokens(df,k):
    top_k = df[(df["is_comment"] == False) & (df["file_target"] == 1.0) & (df["ypred_file"] ==1)]
    top_k = top_k.groupby("filepath").apply(lambda x: x.nlargest(k, "attention_score"), include_groups=True).reset_index(drop=True)
    top_k = top_k[["repo","filepath","line"]].drop_duplicates()
    top_k["flag"] = "topk"
    return top_k


def get_line_level_metrics(df_all):
    sum_line_attn = df_all[(df_all["file_target"] == 1.0) & (df_all["ypred_file"] == 1)]
    missing_files_list = []
    grouped = sum_line_attn.groupby("filepath")
    sorted_df = pd.DataFrame()
    for key, group_df in grouped:
        
        if sum(group_df['is_buggy'] == True) > 0:
            group_df = group_df.sort_values("attention_score", ascending=False)
        else:
            missing_files_list.append(group_df['filepath'].iloc[0])
        sorted_df = pd.concat([sorted_df, group_df])
    del grouped, sum_line_attn
    sorted_df = sorted_df[~sorted_df['filepath'].isin(missing_files_list)]
    sorted_df["order"] = sorted_df.groupby("filepath").cumcount() + 1


    # calculate IFA
    IFA = sorted_df[sorted_df["is_buggy"] == True].groupby("filepath").apply(lambda x: x.nsmallest(1, "order"),include_groups=True).reset_index(drop=True)
    total_true = sorted_df.groupby("filepath").agg({"is_buggy": lambda x: sum(x == "True")}).reset_index()

    # calculate Recall20%LOC
    top_20_percent = sorted_df.groupby("filepath").apply(lambda x: x[x["order"] <= int(0.2 * len(x))]).reset_index(drop=True)
    # Calculate the proportion of buggy lines in the top 20% of lines for each file
    recall20LOC = top_20_percent.groupby("filepath").agg({"is_buggy": "mean"}).reset_index()
    # Rename the 'is_buggy' column to 'recall20LOC'
    recall20LOC.rename(columns={"is_buggy": "recall20LOC"}, inplace=True)

    # Merge with total_true to get the total number of buggy lines for each file
    recall20LOC = recall20LOC.merge(total_true, on="filepath")

    merged_df = sorted_df.merge(total_true, on='filepath', how='left')  # Merge DataFrames

    result = merged_df.groupby('filepath') \
                    .apply(lambda x: x.assign(
                        cummulative_correct_pred=x['is_buggy_x'].eq('True').cumsum(),
                        recall_test=round((x['is_buggy_x'].eq('True').cumsum() / (x['is_buggy_y'] if x['is_buggy_y'] != 0 else 1)), 2), 
                    ))
    len_result = len(result)

    effort20Recall = result[result['recall_test'] <= 0.2].size() # / result.size().sum()


    all_ifa = IFA["order"].tolist()
    all_recall = recall20LOC["recall20LOC"].tolist()
    all_effort = effort20Recall['recall_test'].tolist()
    result_df = pd.DataFrame({"IFA": all_ifa, "Recall@Top20%LOC": all_recall, "Effort@Top20%Recall": all_effort})

    return result_df
