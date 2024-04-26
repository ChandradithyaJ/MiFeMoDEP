import os, re, time, pickle, csv
import pandas as pd
import numpy as np
import matplotlib as plt
import more_itertools
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.utils import compute_class_weight
from DeepLineDP_model import *
from my_utils import *

final_test_pred=pd.read_csv('prediction_DeepLineDP.csv')
all_line_result = pd.DataFrame()

all_repos = final_test_pred['repo'].unique()

tmp_top_k = top_k_tokens(final_test_pred, 150)
merged_df_all = pd.merge(final_test_pred, tmp_top_k, on=["repo","filepath"], how="left")
merged_df_all.loc[merged_df_all["flag"].isna(), "attention_score"] = 0

line_level_result = get_line_level_metrics(merged_df_all)

# printing metrics

line_level_result.boxplot(column= [ "Recall@Top20%LOC", 'Effort@Top20%Recall']) 
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('DeepLineDP') 
plt.show()
