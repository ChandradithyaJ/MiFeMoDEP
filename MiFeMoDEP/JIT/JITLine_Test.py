import pandas as pd
import sklearn.metrics as metrics
import math

path = './'
jitpred_name = 'JITLineReplicationPredictions.csv' 
jitpred_df = pd.read_csv(path+jitpred_name)

# precision, Recall, F1-score, Confusion matrix, False Alarm Rate, Distance-to-Heaven, AUC
prec, rec, f1, _ = metrics.precision_recall_fscore_support(jitpred_df["actual"],jitpred_df["preds"],average='binary') # at threshold = 0.5
tn, fp, fn, tp = metrics.confusion_matrix(jitpred_df["actual"], jitpred_df["preds"], labels=[0, 1]).ravel()
FAR = fp/(fp+tn)
dist_heaven = math.sqrt((pow(1-rec,2)+pow(0-FAR,2))/2.0)
AUC = metrics.roc_auc_score(jitpred_df["actual"], jitpred_df["probs"])

print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1-score: {f1}")
print(f"False Alarm Rate: {FAR}")
print(f"Distance to Heaven: {dist_heaven}")
print(f"AUC: {AUC}")

