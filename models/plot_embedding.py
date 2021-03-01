import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from common.config import *



def plot_embedding(model, detector_file, reduced_dimension, total_beampipe_split):
    assert reduced_dimension == 2 or reduced_dimension == 3
    
    # Create beampipe split df
    bpsplit_data = pd.DataFrame(data={'ordinal_id': np.arange(total_beampipe_split, dtype=int),
                                      'geo_id': np.zeros(total_beampipe_split, dtype=np.uint64),
                                      'volume': [ "Beamline" ] * total_beampipe_split })
    
    # Load node number to geoid encoding
    data = pd.read_csv(detector_file, dtype={'ordinal_id': int, 'geo_id': np.uint64})
    data.drop([0])
    data.index = data.index + total_beampipe_split - 1
    data.loc[:,'ordinal_id'] = data['ordinal_id'].to_numpy() + total_beampipe_split - 1
    
    
    # Combine the two
    data = pd.concat([bpsplit_data, data])
    
    # Color mapping
    color_map = {
        'Beamline':                 '#ffcc00',      # yellow
        'Pixel::NegativeEndcap':    '#000099',      # dark blue
        'Pixel::Barrel':            '#0000ff',      # bright blue
        'Pixel::PositiveEndcap':    '#000099',      # dark blue
        'SStrip::NegativeEndcap':   '#800000',      # dark red
        'SStrip::Barrel':           '#ff0000',      # bright red
        'SStrip::PositiveEndcap':   '#800000',      # dark red
        'LStrip::NegativeEndcap':   '#006600',      # dark green
        'LStrip::Barrel':           '#00cc00',      # bright green
        'LStrip::PositiveEndcap':   '#006600',      # dark green
    }
    
    # Compute embedding
    numbers = data['ordinal_id'].to_numpy()
    embeddings = np.squeeze(model(numbers))
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    logging.info("Computed embeddings, shape: %s",embeddings.shape)

    # Apply TSNE
    if embeddings.shape[1] > reduced_dimension:
        logging.info("Start TSNE dimension reduction (%d -> %d)",embeddings.shape[1],reduced_dimension)
        embeddings = TSNE(n_components=reduced_dimension,n_jobs=20).fit_transform(embeddings)
    
    elif embeddings.shape[1] == reduced_dimension:
        logging.info("No TSNE reduction necessary, target dimension matches embedding dimension")
    
    else:
        logging.error("Embedding dimension not valid!")
        exit(1)
        
    logging.info("Done, shape of reduced embeddings: %s",embeddings.shape)
    
    # Plot
    data['x'] = embeddings[:,0]
    data['y'] = embeddings[:,1]
        
    if reduced_dimension == 2:
        fig = px.scatter(data, x='x', y='y', color='volume', color_discrete_sequence=list(color_map.values()))
        fig.show()
    
    elif reduced_dimension == 3:
        data['z'] = embeddings[:,2]
        fig = px.scatter_3d(data, x='x', y='y', z='z', color='volume', color_discrete_sequence=list(color_map.values()))
        fig.show()
    
    

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=str)
    parser.add_argument("--bpsplit_z",type=int)
    parser.add_argument("--bpsplit_phi",type=int)
    parser.add_argument("--detector",type=str)
    
    options = vars(parser.parse_args())
    
    if options['model_dir'] == None:
        logging.error("Must specify path to model")
        exit()
        
    if options['detector'] == None:
        logging.error("Must specify either 'generic' or 'itk'")
        exit()
        
    if options['bpsplit_z'] == None or options['bpsplit_phi'] == None:
        logging.error("Must specify --bpsplit_z and --bpsplit_phi")
        exit()
        
    detector_file = os.path.join(get_root_dir(), "detectors", options['detector'], "detector_surfaces.csv")
    assert os.path.exists(detector_file)
    assert os.path.exists(options['model_dir'])
    
    model = tf.keras.models.load_model(options['model_dir'], compile=False)
    logging.info("loaded model from '%s'",options['model_dir'])
    
    plot_embedding(model, detector_file, reduced_dimension=3, total_beampipe_split=options['bpsplit_z']*options['bpsplit_phi'])
