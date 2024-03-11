# Train the PDG Encoder NN

import numpy as np
import pandas as pd
import time
from MiFeMoDEP_JIT_Utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# ## Import Datasets and Preprocess
path_to_jit_random = '../DefectorsDataset/defectors/jit_bug_prediction_splits/random'

train_jit_df = pd.read_parquet(f'{path_to_jit_random}/train.parquet.gzip')
train_jit_df = train_jit_df.reset_index(drop=True)

test_jit_df = pd.read_parquet(f'{path_to_jit_random}/test.parquet.gzip')
test_jit_df = test_jit_df.reset_index(drop=True)

train_jit_df['target'] = train_jit_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)
test_jit_df['target'] = test_jit_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)


train_jit_df['content'] = train_jit_df['content'].apply(preprocess_diff)
test_jit_df['content'] = test_jit_df['content'].apply(preprocess_diff)

train_jit_df['code'] = train_jit_df['content'].apply(preprocess_diff_to_code)
test_jit_df['code'] = test_jit_df['content'].apply(preprocess_diff_to_code)

## Uncomment to rebuild
######################################################
# Build the Doc2Vec model
# num_docs = 100
# start = time.time()
# tagged_data = [TaggedDocument(words=_d.split('\n'), tags=[i]) for i, _d in enumerate(train_jit_df['content'][:num_docs])]
# end = time.time()
# print(f'Created tagged data in {end-start} seconds')

# epochs = 100
# d2v_model = Doc2Vec(vector_size=150, window=5, min_count=2, epochs=epochs)
# d2v_model.build_vocab(tagged_data)
# d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=epochs)
# d2v_model.save("./doc2vec_model_train_jit.bin")
#######################################################

d2v_model = Doc2Vec.load("./doc2vec_model_train_jit.bin")

batch_size = 6
num_epochs = 10
learning_rate = 0.01
alpha = 1 # weight of cb_embeddings
beta = 1 # weight of pdg_embeddings

start_idx = 0
stop_idx = 42
upper_bound = 42
modified_train_source_code_df = pd.DataFrame()
data_list = []
for i in range(0, upper_bound): # for loop to save memory
    while start_idx < upper_bound:  
        rows = [i for i in range(start_idx, max(stop_idx, upper_bound))]    
        temp = train_jit_df.iloc[rows]
        temp['pdg_enc_inp'] = None
        start = time.time()

        temp['pdg_enc_inp'] = temp['content'].apply(lambda x : get_PDG_embeddings(d2v_model, x))
        
        print(start_idx, stop_idx, time.time()-start)
        data_list.append(temp)
        start_idx = stop_idx
        stop_idx += 30

if len(data_list) > 0:
    modified_train_jit_df = pd.concat(data_list)
else:
    modified_train_jit_df = data_list[0]


modified_train_jit_df.to_csv('./modified_train_jit_df.csv')
len(modified_train_jit_df)
modified_train_source_code_df = pd.read_csv('./modified_train_source_code_df.csv')

train_dataset = DefectorsTorchDatasetEncoderTraining(modified_train_jit_df)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = NNClassifier()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    start = time.time()
    for X2, y in train_data_loader:
        preds = model(X2)
        loss = loss_fn(preds, y.to(torch.float32).unsqueeze(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch} --> Loss {loss}")        
            
torch.save(enc_model.state_dict(), './MiFeMoDEP_JIT_PDG_Enc.pt')