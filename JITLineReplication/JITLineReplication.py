from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import time, pickle, operator
from lime.lime_tabular import LimeTabularExplainer
import dill
import os

from JITLineUtils import *


def load_and_preprocess_dataset():
    path_to_jit_random = '../DefectorsDataset/defectors/jit_bug_prediction_splits/random'

    train_df = pd.read_parquet(f'{path_to_jit_random}/train.parquet.gzip')
    train_df = train_df.reset_index(drop=True)

    test_df = pd.read_parquet(f'{path_to_jit_random}/test.parquet.gzip')
    test_df = test_df.reset_index(drop=True)


    train_df['target'] = train_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
    test_df['target'] = test_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

    # Extract tokens and Preprocess Data

    train_df['content'] = train_df['content'].apply(preprocess_diff)
    test_df['content'] = test_df['content'].apply(preprocess_diff)

    print("Preprocessed diff files")

    return train_df, test_df

def vectorize_the_diffs(train_df, test_df)
    # dealing only with diff files
    train_git_diff = train_df['content']
    test_git_diff = test_df['content']

    count_vec = CountVectorizer(min_df=3, ngram_range=(1,1))

    X_combined = np.concatenate([train_git_diff, test_git_diff])
    X_combined = count_vec.fit_transform(X_combined)

    X_train = X_combined[:len(train_git_diff)]
    y_train = train_df['target']

    X_test = X_combined[len(train_git_diff):]
    y_test = test_df['target']

    print("Vectorized")

    return X_train, y_train, X_test, y_test

def train_RF_and_predict(X_train, y_train, X_test, y_test, test_df):
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    pickle.dump(clf, open('./JITLineReplicationClassifier.pkl', 'wb'))
    print(f'Training RF Classifier finished in {end-start} secs')

    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)

    creating a predictions dataframe for analysis
    pred_df = pd.DataFrame()
    pred_df['probs'] = probs[:, 1]
    pred_df['preds'] = preds
    pred_df['actual'] = y_test

    concatenate with buggy lines list
    pred_df = pd.concat([pred_df, test_df[['lines', 'commit']]], axis=1)

    pred_df.to_csv('./JITLineReplicationPredictions.csv')

def LIME_for_line_level_scores(X_train, X_test):
# Local Interpretable Model-Agnostic Explanations (LIME) for Ranking Defective Lines

    num_features = X_train.shape[1]
    top_k_tokens = np.arange(10,201,10)
    agg_methods = ['avg','median','sum']
    max_str_len_list = 100
    max_tokens = 100
    line_score_df_col_name = ['total_tokens', 'line_level_label'] + ['token'+str(i) for i in range(1,max_str_len_list+1)] + [agg+'-top-'+str(k)+'-tokens' for agg in agg_methods for k in top_k_tokens] + [agg+'-all-tokens' for agg in agg_methods]

    def get_LIME_Explainer(X_train, reinit=False):
        feature_names = [str(i) for i in range(num_features)]
        LIME_explainer_path = './JITLineReplicationLIME.pkl'
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
        
        pred_df = pd.read_csv('./JITLineReplicationPredictions.csv')
        pred_df = pd.concat([test_df['commit'], pred_df], axis=1)
        # empty list initialization
        pred_df['predicted_buggy_lines'] = pred_df['lines'].apply(lambda x : [])
        
        line_level_df = pd.read_csv('line_level_jit_test.csv')
        
        # for each row, predict buggy lines
        prev_idx = -1 # deleting checkpoints
        # takes around a minute to process buggy files
        for i in range(len(pred_df)):
            
            if i%10 == 0:
                print(i)

            start = time.time()
            
            if pred_df['preds'][i] == 0 or pred_df['actual'][i] == 0:
                continue
                
            code_change_from_line_level_df = list(line_level_df[line_level_df['commit_hash']==pred_df['commit'][i]]['code_change'])
            line_level_label = list(line_level_df[line_level_df['commit_hash']==pred_df['commit'][i]]['is_buggy_line'])
            
            line_score_df = pd.DataFrame(columns = line_score_df_col_name)  
            line_score_df = line_score_df.set_index('line_num')
            
            exp = explainer.explain_instance(np.squeeze(X_test[i].toarray()),
                                            clf.predict_proba,
                                            num_features=num_features,
                                            top_labels=1,
                                            num_samples=5000
                                            )
        
            sorted_feature_score_dict, tokens_list = preprocess_feature_from_explainer(exp)
            
            for line_num, line in enumerate(code_change_from_line_level_df):
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
            if os.path.exists(f"./JITLine_LineLevelResult_{prev_idx}.pkl"):
                os.remove(f"./JITLine_LineLevelResult_{prev_idx}.pkl")
            pickle.dump(all_buggy_line_result_df, open(f"./JITLine_LineLevelResult_{i}.pkl", 'wb'))
            prev_idx = i
            print(f"Finished calculating line scores of file idx {i} in {time.time()-start}s")
            
        return all_buggy_line_result_df
    
    # calculate the scores for each line
    clf = pickle.load(open('./JITLineReplicationClassifier.pkl', 'rb'))
    explainer = get_LIME_Explainer(X_train, reinit=True)
    line_level_result = eval_with_LIME(clf, explainer, X_test)
    pickle.dump(line_level_result, open('./JITLine_LineLevelResult.pkl', 'wb'))


if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_dataset()
    X_train, y_train, X_test, y_test = vectorize_the_diffs(train_df, test_df)
    train_RF_and_predict(X_train, y_train, X_test, y_test, test_df)
    LIME_for_line_level_scores(X_train, X_test)