import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import dill
import os, pickle, time, re, operator
import warnings
warnings.filterwarnings('ignore')

X_train = pickle.load(open('./X_train.pkl', 'rb'))
X_test = pickle.load(open('./X_test.pkl', 'rb'))
test_pred_df = pd.read_csv('./balanced_500_test_source_code.csv').drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

num_features = X_train.shape[1]
top_k_tokens = np.arange(10,201,10)
agg_methods = ['avg','median','sum']
max_str_len_list = 100
max_tokens = 100
line_score_df_col_name = ['total_tokens', 'line_level_label', 'line_num'] + ['token'+str(i) for i in range(1,max_str_len_list+1)] + [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]

def get_LIME_Explainer(X_train, reinit=False):
    feature_names = [str(i) for i in range(num_features)]
    LIME_explainer_path = './LIME_for_MiFeMoDEP.pkl'
    class_names = ['not defective', 'defective']
    
    if not os.path.exists(LIME_explainer_path) or reinit:
        start = time.time()
        explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, discretize_continuous=False, random_state=42)
        print(f'Finished training LIME in {time.time()-start}s')
        dill.dump(explainer, open(LIME_explainer_path, 'wb'))
    else:
        explainer = dill.load(open(LIME_explainer_path, 'rb'))
    
    return explainer

def eval_with_LIME(clf, explainer, test_features):
    
    def preprocess_feature_from_explainer(exp):
        features_val = exp.as_list(label=1)
        new_features_val = [tup for tup in features_val if float(tup[1]) > 0]

        feature_dict = {re.sub('\s.*','',val[0]):val[1] for val in new_features_val}

        sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1), reverse=True)

        sorted_feature_dict = {tup[0]:tup[1] for tup in sorted_feature_dict}
        tokens_list = list(sorted_feature_dict.keys())

        return sorted_feature_dict, tokens_list
    
    def add_agg_scr_to_list(line_stuff, scr_list):
        if len(scr_list) < 1:
            scr_list.append(0)

        line_stuff.append(np.mean(scr_list))
        line_stuff.append(np.median(scr_list))
        line_stuff.append(np.sum(scr_list))
        
    all_buggy_line_result_df = []
    
    # empty list initialization
    test_pred_df['predicted_buggy_lines'] = test_pred_df['lines'].apply(lambda x : [])
    line_level_df = pd.read_csv('./JITLineReplicationForMiFeMoDEP/line_level_jit_test.csv').drop(['Unnamed: 0'], axis=1)
    
    # for each row, predict buggy lines
    prev_idx = -1 # deleting checkpoints
    # takes around a minute to process buggy files
    for i in range(len(test_pred_df)):
        
        if i%10 == 0:
            print(i)

        start = time.time()
        
        if test_pred_df['y_pred'][i] == 0 or test_pred_df['target'][i] == 0:
            continue
            
        # get the required code lines and labels from the line level dataset
        source_code_lines = list(line_level_df[line_level_df['commit_hash']==test_pred_df['commit'][i]]['code_change'])
        line_level_label = list(line_level_df[line_level_df['commit_hash']==test_pred_df['commit'][i]]['is_buggy_line'])
        
        line_score_df = pd.DataFrame(columns=line_score_df_col_name)  
        line_score_df = line_score_df.set_index('line_num')

        exp = explainer.explain_instance(X_test[i],
                                         clf.predict_proba,
                                         num_features=num_features,
                                         top_labels=1,
                                         num_samples=5000
                                        )
    
        sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(exp)
        
        for line_num, line in enumerate(source_code_lines):
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
        
        line_score_df['commit_id'] = [i]*len(line_level_label)
        line_score_df['line_level_label'] = line_level_label

        all_buggy_line_result_df.append(line_score_df)
        
        del exp, sorted_feature_score_dict, tokens_list, line_score_df
        if os.path.exists(f"./MiFeMoDEP_LineLevelResult_{prev_idx}.pkl"):
            os.remove(f"./MiFeMoDEP_LineLevelResult_{prev_idx}.pkl")
        pickle.dump(all_buggy_line_result_df, open(f"./MiFeMoDEP_LineLevelResult_{i}.pkl", 'wb'))
        prev_idx = i
        print(f"Finished calculating line scores of file idx {i} in {time.time()-start}s")
    
    if os.path.exists(f"./MiFeMoDEP_LineLevelResult_{prev_idx}.pkl"):
        os.remove(f"./MiFeMoDEP_LineLevelResult_{prev_idx}.pkl")
    return all_buggy_line_result_df


def eval_line_level():
    clf = pickle.load(open('./MiFeMoDEP_SourceCode_RF.pkl', 'rb'))
    
    explainer = get_LIME_Explainer(X_train, reinit=True)
    
    line_level_result = eval_with_LIME(clf, explainer, X_test)
    
    pickle.dump(line_level_result, open('./MiFeMoDEP_LineLevelResult.pkl', 'wb'))

if __name__ == "__main__":
    eval_line_level()