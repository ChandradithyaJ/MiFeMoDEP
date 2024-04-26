import pandas as pd
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

path_to_node_data = "./"
node_df = pd.read_csv(path_to_node_data+"nodes_all_graphs.csv")

# Build the Doc2Vec model
num_docs = 42
start = time.time()
tagged_data = [TaggedDocument(words=_d, tags=[i]) for i, _d in enumerate(node_df['dictionary'][:num_docs])]
end = time.time()

print(f'Created tagged data in {end-start} seconds')

# train a Doc2Vec Model
epochs = 100

d2v_model = Doc2Vec(vector_size=150, window=5, min_count=2, epochs=epochs)
d2v_model.build_vocab(tagged_data)
d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=epochs)
d2v_model.save("./doc2vec_nodel_nodes_from_graphs.bin")