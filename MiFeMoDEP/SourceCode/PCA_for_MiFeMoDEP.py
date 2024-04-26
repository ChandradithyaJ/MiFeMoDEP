from sklearn.decomposition import PCA
import torch, numpy as np
import h5py, time, pickle

num_features = 1450000

def PCA_fit(filepath_to_embeds):
    with h5py.File(filepath_to_embeds, "r") as f:
        dataset = f["all_cb_embeds"]
        all_cb_embeds = dataset[:] # numpy arrays

    num_embeddings = all_cb_embeds.shape[0]
    all_cb_embeds = all_cb_embeds.reshape(num_embeddings, -1)
    print(all_cb_embeds.shape)

    # fit the PCA model on a section of the data
    num_embeddings_for_fitting = 128
    all_cb_embeds = all_cb_embeds[:num_embeddings_for_fitting, :num_features]

    pca = PCA(n_components=128)

    print("Starting PCA fitting...")
    start = time.time()
    pca_data = pca.fit_transform(all_cb_embeds)
    print(f"PCA trained in {time.time()-start}s")
    print(pca_data.shape, type(pca_data))
    pickle.dump(pca, open('MiFeMoDEP_PCA.pkl', 'wb'))

def PCA_infer(filepath_to_embeds):
    pca = pickle.load(open('MiFeMoDEP_PCA.pkl', 'rb'))
    with h5py.File(filepath_to_embeds, "r") as f:
        dataset = f["all_cb_embeds"]
        all_cb_embeds = dataset[:]
    
    num_embeddings = all_cb_embeds.shape[0]
    all_cb_embeds = all_cb_embeds.reshape(num_embeddings, -1)
    print("Number of embeddings: ", num_embeddings)

    all_cb_embeds_transformed = np.empty((0, 128))
    i = 0
    while i < num_embeddings:
        transform_cb_embeds = all_cb_embeds[i:i+128, :num_features]

        print(f"Starting PCA transforming for batch {int(i/128)+1}")
        start = time.time()
        transform_cb_embeds = pca.transform(transform_cb_embeds)
        print(f"Batch {int(i/128)+1} data transformed in {time.time()-start}s")
        print(transform_cb_embeds.shape)
        all_cb_embeds_transformed = np.append(all_cb_embeds_transformed, transform_cb_embeds, axis=0)

        i += 128

    del all_cb_embeds, i

    if filepath_to_embeds == "./JITLineReplicationForMiFeMoDEP/all_cb_embeds_2.h5":
        with h5py.File("all_test_cb_embeds_transformed.h5", "r") as f:
            dataset = f["all_cb_embeds_transformed"]
            all_cb_embeds_1 = dataset[:]
        all_cb_embeds_transformed = np.vstack((all_cb_embeds_1, all_cb_embeds_transformed))
        with h5py.File("all_test_cb_embeds_transformed.h5", "w") as f:
            f.create_dataset("all_cb_embeds_transformed", data=all_cb_embeds_transformed)
    else:
        with h5py.File("all_test_cb_embeds_transformed.h5", "w") as f:
            f.create_dataset("all_cb_embeds_transformed", data=all_cb_embeds_transformed)

def PCA_single_input(pca, cb_embeds, num_features):
    transform_cb_embeds = pca.transform(cb_embeds[:, :num_features])
    return transform_cb_embeds

if __name__ == "__main__":
    # PCA_fit("./all_cb_embeds.h5")
    # PCA_infer("./all_test_cb_embeds.h5")
