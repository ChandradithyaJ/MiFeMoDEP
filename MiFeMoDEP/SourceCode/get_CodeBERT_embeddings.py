import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, AutoModel
import h5py, pandas as pd, time

cb_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
cb_model = AutoModel.from_pretrained('microsoft/codebert-base')
max_len = 10000

# CodeBERT
def get_CodeBERT_context_embeddings(tokenizer, model, source_code, max_len):
    tokens = tokenizer.tokenize(source_code)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    # if there are more than L tokens give them in batches
    L = 512 # token stream size
    N0 = 64 # overlap factor for context preservation
    batch_num = 0

    tensor_embeddings = []
    if len(tokens_ids) > L:
        i = 0
        while i + N0 < len(tokens_ids):
            if batch_num == max_len:
                break

            tokens_batch = tokens_ids[i : i + L]
            with torch.no_grad():
                embeddings = model(torch.tensor(tokens_batch).to(torch.int64)[None,:])[0]
            tensor_embeddings.append(embeddings)
            i += (L - N0)
            batch_num += 1
    else:
        with torch.no_grad():
            tensor_embeddings = model(torch.tensor(tokens_ids).to(torch.int64)[None,:])[0]

    # concatenate
    if len(tokens_ids) > 512:
        tensor_embeddings = torch.concat(tensor_embeddings, dim=1)

    if tensor_embeddings.size(1) > max_len:
        tensor_embeddings = tensor_embeddings[:, :max_len, :]
    else:
        padding = torch.zeros((1, max_len - tensor_embeddings.size(1), 768))
        tensor_embeddings = torch.cat((tensor_embeddings, padding), dim=1)

    tensor_embeddings = torch.squeeze(tensor_embeddings, dim=0)
    return tensor_embeddings

"""
Uncomment the below as required to get the respective CodeBERT embeddings
"""

# train_source_code_df = pd.read_csv('preprocessed_train_source_code.csv')
# train_source_code_df = train_source_code_df.reset_index(drop=True)

# random_1200_balanced_df = pd.read_csv('random_1200_balanced.csv')
# random_1200_balanced_df = random_1200_balanced_df.reset_index(drop=True)

# start = time.time()
# train_source_code_df = train_source_code_df.merge(random_1200_balanced_df, on=['repo', 'commit', 'filepath'], how='inner')
# train_source_code_df.to_csv('./random_1200_balanced_train_with_content.csv')
# print(f'Created and saved balanced dataset in {time.time()-start}s')
# del random_1200_balanced_df
# print(len(train_source_code_df))

# train_source_code_df = pd.read_csv('./random_1200_balanced_train_with_content.csv')
# train_source_code_df = train_source_code_df.reset_index(drop=True)

# test_source_code_df = pd.read_csv('./JITLineReplicationForMiFeMoDEP/preprocessed_test_source_code.csv')
# test_source_code_df_1 = test_source_code_df[test_source_code_df['target'] == 1].sample(250, random_state=42)
# test_source_code_df_0 = test_source_code_df[test_source_code_df['target'] == 0].sample(250, random_state=42)
# test_source_code_df = pd.concat([test_source_code_df_1, test_source_code_df_0], ignore_index=True)
# test_source_code_df.to_csv('./balanced_500_test_source_code.csv')
# print('Created Balanced Test Dataset')

def create_embeddings():

    all_cb_embeds = torch.tensor([])
    """
    Uncomment the lines below if you're processing in batches of h5py files
    """
    # with h5py.File("all_cb_embeds_2.h5", "r") as f:
    #     dataset = f["all_cb_embeds"]
    #     all_cb_embeds = dataset[:]
    #     all_cb_embeds = torch.from_numpy(all_cb_embeds)

    for index, row in test_source_code_df.iterrows():
        if type(row['content']) == float:
            row['content'] = " "
        start = time.time()
        cb_embeds = get_CodeBERT_context_embeddings(cb_tokenizer, cb_model, row['content'])
        all_cb_embeds = torch.cat((all_cb_embeds, cb_embeds.unsqueeze(0)), dim=0)
        del cb_embeds
        print(index, time.time()-start)
        if index%20 == 0:
            with h5py.File("all_test_cb_embeds.h5", "w") as f:
                f.create_dataset("all_cb_embeds", data=all_cb_embeds)

    with h5py.File("all_test_cb_embeds.h5", "w") as f:
        f.create_dataset("all_cb_embeds", data=all_cb_embeds)

if __name__ == "__main__":
	create_embeddings()
