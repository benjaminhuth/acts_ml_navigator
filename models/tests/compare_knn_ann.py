import numpy as np
import pandas as pd
import logging
import os
import timeit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex


logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)


root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
embedding_model_file = root_dir + "models/embeddings/20201126-163733-emb50-acc40"
detector_data_file = root_dir + "detector/detector_surfaces.csv"


# Load model
embedding_encoder = tf.keras.models.load_model(embedding_model_file, compile=False)
logging.info("loaded model from '%s'",embedding_model_file)

embedding_shape = embedding_encoder(0).shape
logging.info("embedding shape: %s",embedding_shape)

# Get all valid nodes
detector_data = pd.read_csv(detector_data_file, dtype={'geo_id': np.uint64})
all_geo_ids = detector_data['geo_id'].to_numpy()
all_numbers = detector_data['ordinal_id'].to_numpy()

# Transformation dictionary
geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }
logging.info("loaded detector data from '%s'", detector_data_file)

# Establish neighbouring index
all_embeddings = np.squeeze(embedding_encoder(all_numbers).numpy())
nn = NearestNeighbors()
nn.fit(all_embeddings)

ann = AnnoyIndex(embedding_shape[0], 'euclidean')
for i, emb in zip(all_numbers, all_embeddings):
    ann.add_item(i,emb)
    
ann.build(2000)
    
logging.info("built neighbouring index")

dim_max = np.amax(all_embeddings,axis=0)
dim_min = np.amin(all_embeddings,axis=0)

num_vec = 1000
random_vectors = [ np.random.uniform(low, high, num_vec) for low, high in zip(dim_min, dim_max) ]

random_vectors = np.vstack(random_vectors).transpose()

score = 0.0

for vec in random_vectors:
    ten_nn = nn.kneighbors(vec.reshape(1,-1), 10, return_distance=False).flatten()
    ten_ann = np.array(ann.get_nns_by_vector(vec, 10, search_k=-1, include_distances=False))
    
    combinded = np.unique(np.concatenate([ ten_nn, ten_ann ]))
    
    score += (len(combinded) - 10) / 10
    
score /= len(random_vectors)

logging.info("score: %f", score)

iterations = 1

t_0 = timeit.default_timer()

nn.kneighbors(random_vectors, 10, return_distance=False)

t_1 = timeit.default_timer()

logging.info("sklearn time: %f",(t_1-t_0)/iterations)


t_0 = timeit.default_timer()

for vec in random_vectors:
    ann.get_nns_by_vector(vec, 10, search_k=-1, include_distances=False)
    
t_1 = timeit.default_timer()

logging.info("annoy time: %f",(t_1-t_0)/iterations)




