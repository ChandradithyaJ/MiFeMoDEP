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
from DeepLineDP_model import *
from my_utils import *

# ## Loading Data
path_to_line_random = '.'

train_df = pd.read_parquet(f'{path_to_line_random}/train.parquet.gzip')
train_df = train_df.reset_index(drop=True)

test_df = pd.read_parquet(f'{path_to_line_random}/test.parquet.gzip')
test_df = test_df.reset_index(drop=True)

train_df['target'] = train_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
test_df['target'] = test_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

train_df_1 = train_df[train_df['target'] == 1].sample(75, random_state=42)
train_df_0 = train_df[train_df['target'] == 0].sample(75, random_state=42)

# Combine the DataFrames
train_df = pd.concat([train_df_1, train_df_0], ignore_index=True)

data_root_dir = '../datasets/original/'  #a list of code2d from prepare_code2d()
save_dir = "../datasets/preprocessed_data/"

char_to_remove = ['+','-','*','/','=','++','--','\\','<str>','<char>','|','&','!']

def is_comment_line(code_line, comments_list):
    '''
        input
            code_line (string): source code in a line
            comments_list (list): a list that contains every comments
        output
            boolean value
    '''

    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('#'):
        return True
    elif code_line in comments_list:
        return True

    return False

def is_empty_line(code_line):
    '''
        input
            code_line (string)
        output
            boolean value
    '''

    if len(code_line.strip()) == 0:
        return True

    return False

def preprocess_code_line(code_line):
    '''
        input
            code_line (string)
    '''

    code_line = re.sub("\'\'", "\'", code_line)
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = re.sub('\b\d+\b','',code_line)
    code_line = re.sub("\\[.*?\\]", '', code_line)
    code_line = re.sub("[\\.|,|:|;|{|}|(|)]", ' ', code_line)

    for char in char_to_remove:
        code_line = code_line.replace(char,' ')

    code_line = code_line.strip()

    return code_line

def preprocess_code(code_str):
    '''
        input
            code_str (multi line str)
    '''
    if(code_str is None):
        return ''
    code_str = code_str.decode("latin-1")
    code_lines = code_str.splitlines()

    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []

    # multi-line comments
    comments = re.findall(r'("""(.*?)""")|(\'\'\'(.*?)\'\'\')', code_str, re.DOTALL)
    comments_temp = []
    for tup in comments:
        temp = ''
        for s in tup:
            temp += s
        comments_temp.append(temp)
    comments_str = '\n'.join(comments_temp)
    comments_list = comments_str.split('\n')

    for l in code_lines:
        l = l.strip()
        is_comment = is_comment_line(l,comments_list)
        is_comments.append(is_comment)

        if not is_comment:
            l = preprocess_code_line(l)

        preprocess_code_lines.append(l)

    return ' \n '.join(preprocess_code_lines)

# preprocessing on train and test dataframes

train_df['content'] = train_df['content'].apply(preprocess_code)
test_df['content'] = test_df['content'].apply(preprocess_code)
train_df.to_csv('./preprocessed_train_df.csv')
test_df.to_csv('./preprocessed_test_df.csv')
