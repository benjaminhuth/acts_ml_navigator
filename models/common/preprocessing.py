import os
import logging

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors

class GeoidNumberConverter:
    '''
    Provides conversion between the GeoID-representation and the ordinal-number-representation of Acts surfaces.
    '''
    def __init__(self, detector_file):
        detector_data = pd.read_csv(detector_file, dtype={'geo_id': np.uint64})
        self.all_geo_ids = detector_data['geo_id'].to_numpy()
        self.all_numbers = detector_data['ordinal_id'].to_numpy()
        
        self.geoid_to_number = { geo_id : number for geo_id,number in zip(self.all_geo_ids, self.all_numbers) }
        self.number_to_geoid = { number : geo_id for geo_id,number in zip(self.all_geo_ids, self.all_numbers) }
        
    def get_all_numbers(self):
        return self.all_numbers
    
    def get_all_geoids(self):
        return self.all_geo_ids
    
    def to_numbers(self, geoid_array):
        return np.array([ self.geoid_to_number[geoid] for geoid in geoid_array ])
    
    def to_geoids(self, number_array):
        return np.array([ self.number_to_geoid[number] for number in number_array ])
    
    
def extract_embedding_models(embedding_dir):
    '''
    returns list of tuples (path, embedding_dim, score)
    '''
    embedding_models = []
    
    for entry in os.scandir(embedding_dir):
        if entry.is_dir():
            parts = entry.name.split('-')
            
            if len(parts) != 4:
                continue
            
            dim = int(parts[2][3:])
            acc = int(parts[3][3:])
            
            embedding_models.append( (entry.path, dim, acc) )
    
    return embedding_models


def extract_propagation_data(propagation_dir):
    '''
    returns list of tuples (path, num_simulated_acts_events)
    '''
    propagation_data = []
    
    for entry in os.scandir(propagation_dir):
        if entry.is_file():
            filename_no_extension = entry.name[:-4]
            parts = filename_no_extension.split('-')
            
            if len(parts) != 4:
                continue
            
            num_events = int(parts[3][1:])
            
            propagation_data.append( (entry.path, num_events) )
    
    return propagation_data
    
    


def prepare_data(propagation_file, embedding_file, detector_file):
    '''
    returns ( x_embs, x_pars, y_embs, nearest-neighbor-index )
    
    * x_embs, x_pars, y_embs is a list of tracks. 
    * x_embs.shape = (num_hits,embedding_dim)
    * x_pars.shape = (num_hits,num_params)
    * y_embs.shape = (num_hits,embedding_dim)
    '''
    
    ###########################
    # Import propagation_data #
    ###########################
    
    propagation_data = pd.read_csv(propagation_file, dtype={'start_id': np.uint64, 'end_id': np.uint64})
    
    sep_idxs = propagation_data[propagation_data['start_id'] == 0].index.to_numpy()
    sequence_lengths = np.diff(sep_idxs)
    logging.info("Imported %d tracks, the maximum sequence length is %d",len(sep_idxs),np.max(sequence_lengths))
   
    x_geoids = propagation_data['start_id'].to_numpy()
    x_pars = propagation_data[['dir_x', 'dir_y', 'dir_z', 'qop']].to_numpy().astype(np.float32)
    y_geoids = propagation_data['end_id'].to_numpy()
    
    ########################
    # Convert to embedding #
    ########################
    
    converter = GeoidNumberConverter(detector_file)
    embedding_encoder = tf.keras.models.load_model(embedding_file, compile=False)
    
    x_embs = np.squeeze(embedding_encoder(converter.to_numbers(x_geoids)))
    y_embs = np.squeeze(embedding_encoder(converter.to_numbers(y_geoids)))
    
    embedding_dim = x_embs.shape[1]
    all_embeddings = np.squeeze(embedding_encoder(converter.get_all_numbers()).numpy())
    
    nn = NearestNeighbors()
    nn.fit(all_embeddings)
    
    logging.info("imported training data and built neighbouring index")
    logging.info("x_embs   shape: %s,\tdtype: %s", x_embs.shape, x_embs.dtype)
    logging.info("x_params shape: %s,\tdtype: %s", x_pars.shape, x_pars.dtype)
    logging.info("y_embs   shape: %s,\tdtype: %s", y_embs.shape, y_embs.dtype)
    logging.info("training data size: %f MB", (x_pars.nbytes + y_embs.nbytes) / 1.e6)
    
    ######################
    # Group to sequences #
    ######################
    
    x_emb_seqs = []
    x_par_seqs = []
    y_emb_seqs = []
    
    for i in range(len(sep_idxs)-1):
        x_emb_seqs.append(x_embs[sep_idxs[i]:sep_idxs[i+1]])
        x_par_seqs.append(x_pars[sep_idxs[i]:sep_idxs[i+1]])
        y_emb_seqs.append(y_embs[sep_idxs[i]:sep_idxs[i+1]])
        
    x_emb_seqs.append(x_embs[sep_idxs[-1]:])
    x_par_seqs.append(x_pars[sep_idxs[-1]:])
    y_emb_seqs.append(y_embs[sep_idxs[-1]:])
    
    assert len(x_emb_seqs) == len(x_par_seqs) == len(y_emb_seqs)
    logging.info("grouped training data into tracks")
    
    return x_emb_seqs, x_par_seqs, y_emb_seqs, nn
