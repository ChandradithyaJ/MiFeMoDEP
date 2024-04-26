from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import time, pickle, math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path_to_MiFeMoDEP_results = "./MiFeMoDEP_LineLevelResult.pkl"
with open(path_to_MiFeMoDEP_results,"rb") as f:
  list_df = pickle.load(f)

top_k_tokens = np.arange(10,201,10)
agg_methods = ['avg','median','sum']
score_cols = [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]
line_score_df_col_name = ['commit_id', 'total_tokens', 'line_level_label'] + score_cols

def get_line_level_metrics(line_score,label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1)) # cannot pass line_score as list T-T
    pred = np.round(line_score)

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val.item()) for val in line_score]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr',ascending=False)
    line_df['row'] = np.arange(1, len(line_df)+1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0

    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2*len(line_df))

    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row']-1
        label_list = line_df['label'].values.tolist()

        all_rows = len(label_list)

        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows])/len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10])/len(label_list[:10])

        # find recall
        LOC_20_percent = line_df.head(int(0.2*len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num))/float(len(real_buggy_lines))

        # find effort @20% LOC recall
        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc

def create_tmp_df(all_commits,agg_methods):
    df = pd.DataFrame(columns = ['commit_id']+agg_methods)
    df['commit_id'] = all_commits
    df = df.set_index('commit_id')
    return df

def eval_line_level_at_commit(file_df):
    RF_result = file_df
    RF_result = RF_result[line_score_df_col_name]

    all_commits = RF_result['commit_id'].unique().tolist()

    IFA_df = create_tmp_df(all_commits, score_cols)
    recall_20_percent_effort_df = create_tmp_df(all_commits, score_cols)
    effort_20_percent_recall_df = create_tmp_df(all_commits, score_cols)
    precision_df = create_tmp_df(all_commits, score_cols)
    recall_df = create_tmp_df(all_commits, score_cols)
    f1_df = create_tmp_df(all_commits, score_cols)
    AUC_df = create_tmp_df(all_commits, score_cols)
    top_10_acc_df = create_tmp_df(all_commits, score_cols)
    MCC_df = create_tmp_df(all_commits, score_cols)
    bal_ACC_df = create_tmp_df(all_commits, score_cols)

    for commit in all_commits:
        IFA_list = []
        recall_20_percent_effort_list = []
        effort_20_percent_recall_list = []
        top_10_acc_list = []

        cur_RF_result = RF_result[RF_result['commit_id']==commit]

        to_save_df = cur_RF_result[['commit_id',  'total_tokens',  'line_level_label',  'sum-all-tokens']].copy()

        scaler = MinMaxScaler()
        line_score = scaler.fit_transform(np.array(to_save_df['sum-all-tokens']).reshape(-1, 1))
        # Assign directly to the entire column
        to_save_df['line_score'] =  [x.item() for x in line_score].copy()

        to_save_df = to_save_df.drop(['sum-all-tokens','commit_id'], axis=1)
        to_save_df = to_save_df.sort_values(by='line_score', ascending=False)
        to_save_df['row'] = np.arange(1,len(to_save_df)+1)
        # to_save_df.to_csv('./data/line-level_ranking_result/_'+str(commit)+'.csv',index=False)

        line_label = cur_RF_result['line_level_label'].values.tolist()

        for n, agg_method in enumerate(score_cols):

            RF_line_scr = cur_RF_result[agg_method].values.tolist()

            IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc = get_line_level_metrics(RF_line_scr, line_label) # get the metrics for each aggregate method

            IFA_list.append(IFA)
            recall_20_percent_effort_list.append(top_20_percent_LOC_recall)
            effort_20_percent_recall_list.append(effort_at_20_percent_LOC_recall)
            top_10_acc_list.append(top_10_acc)

        IFA_df.loc[commit] = IFA_list
        recall_20_percent_effort_df.loc[commit] = recall_20_percent_effort_list
        effort_20_percent_recall_df.loc[commit] = effort_20_percent_recall_list
        top_10_acc_df.loc[commit] = top_10_acc_list

    return to_save_df,IFA_df,recall_20_percent_effort_df,effort_20_percent_recall_df,top_10_acc_df

def save_df(root,name,df):
  df.to_csv(root+"/"+name+".csv")

def plot_result(df):
    fig, axs = plt.subplots(1,4, figsize=(20,5))
    metrics = ['top_10_acc', 'recall_20_percent_effort','effort_20_percent_recall', 'IFA']
    metrics_label = ['Top-10-ACC', 'Recall20%Effort', 'Effort@20%LOC', 'IFA']
    to_save,ifa,recall_20_percent_effort,effort_20_percent_recall, top_10_acc = eval_line_level_at_commit(df)
    root = "./"
    save_df(root,"to_save",to_save)
    map = {'top_10_acc': top_10_acc,'recall_20_percent_effort': recall_20_percent_effort, 'effort_20_percent_recall':effort_20_percent_recall,'IFA':ifa,'to_save':to_save}
    for i in range(0,4):
        save_df(root,metrics[i],map[metrics[i]])
        # result_df = pd.read_csv('./text_metric_line_eval_result/'+cur_proj+'_'+metrics[i]+'_min_df_3_300_trees.csv')
        result_df = map[metrics[i]]
        result = result_df['sum-all-tokens']
        axs[i].boxplot(result)
        axs[i].set_xticklabels([metrics_label[i]])
        axs[i].tick_params(axis='x', which='major', labelsize=15)
        axs[i].tick_params(axis='y', which='major', labelsize=12)

    plt.show()


if __name__ = "__main__":
    df = pd.concat(list_df)
    plot_result(df)