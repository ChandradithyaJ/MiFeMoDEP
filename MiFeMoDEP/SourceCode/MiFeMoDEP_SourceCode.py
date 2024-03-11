import pandas as pd
import time, ast
from MiFeMoDEP_SourceCode_Utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Import Datasets and Preprocess

path_to_source_code_random = '../DefectorsDataset/defectors/line_bug_prediction_splits/random'

train_source_code_df = pd.read_parquet(f'{path_to_source_code_random}/train.parquet.gzip')
train_source_code_df = train_source_code_df.reset_index(drop=True)

train_source_code_df['target'] = train_source_code_df['lines'].apply(lambda line : 0 if len(line) == 0 else 1)

train_source_code_df['content'] = train_source_code_df['content'].apply(lambda x : '' if x is None else x.decode("latin-1"))

## Uncomment to rebuild
###############################################
# Build the Doc2Vec model
# num_docs = 42
# start = time.time()
# tagged_data = [TaggedDocument(words=preprocess_source_code_for_d2v(_d), tags=[i]) for i, _d in enumerate(train_source_code_df['content'][:num_docs])]
# end = time.time()

# print(f'Created tagged data in {end-start} seconds')

# # train a Doc2Vec Model
# epochs = 100

# d2v_model = Doc2Vec(vector_size=150, window=5, min_count=2, epochs=epochs)
# d2v_model.build_vocab(tagged_data)
# d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=epochs)
# d2v_model.save("./doc2vec_model_train_source_code.bin")
#################################################

d2v_model = Doc2Vec.load("./doc2vec_model_train_source_code.bin")

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
        rows = [i for i in range(start_idx, max(stop_idx, upper_bound)) if i != 4]    
        temp = train_source_code_df.iloc[rows]
        temp['pdg_enc_inp'] = None
        start = time.time()

        temp['pdg_enc_inp'] = temp['content'].apply(lambda x : get_PDG_embeddings(d2v_model, x))
        
        print(start_idx, stop_idx, time.time()-start)
        data_list.append(temp)
        start_idx = stop_idx
        stop_idx += 30

if len(data_list) > 0:
    modified_train_source_code_df = pd.concat(data_list)
else:
    modified_train_source_code_df = data_list[0]

modified_train_source_code_df.to_csv('./modified_train_source_code_df.csv')
len(modified_train_source_code_df)
modified_train_source_code_df = pd.read_csv('./modified_train_source_code_df.csv')

train_dataset = DefectorsTorchDatasetEncoderTraining(modified_train_source_code_df)
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
            
torch.save(enc_model.state_dict(), './MiFeMoDEP_SourceCode_PDG_Enc.pt')