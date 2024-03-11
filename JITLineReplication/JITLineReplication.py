from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd
import time, pickle

from JITLineUtils import *

# get data
path_to_jit_random = '../DefectorsDataset/defectors/jit_bug_prediction_splits/random'
train_df = pd.read_parquet(f'{path_to_jit_random}/train.parquet.gzip')
train_df = train_df.reset_index(drop=True)
test_df = pd.read_parquet(f'{path_to_jit_random}/test.parquet.gzip')
test_df = test_df.reset_index(drop=True)

# create file level labels
train_df['target'] = train_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
test_df['target'] = test_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

# preprocess the diff
train_df['content'] = train_df['content'].apply(preprocess_diff)
test_df['content'] = test_df['content'].apply(preprocess_diff)

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

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
pickle.dump(clf, open('./JITLineReplicationClassifier.pkl', 'wb'))
print(f'Training RF Classifier finished in {end-start} secs')

probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)

# creating a predictions dataframe for analysis
pred_df = pd.DataFrame()
pred_df['probs'] = probs[:, 1]
pred_df['preds'] = preds
pred_df['actual'] = y_test

# concatenate with buggy lines list
pred_df = pd.concat([pred_df, test_df['lines']], axis=1)

pred_df.to_csv('./JITLineReplicationPredictions.csv')