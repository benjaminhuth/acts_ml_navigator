import os
import sys
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utility.data_import import get_sorted_model_dirs


def main(argv):
    # Sort model subdirs by accuracy
    #model_dir = '../data/embeddings/' + get_sorted_model_dirs('../data/embeddings/')[0]
    model_dir = '../data/embeddings/20201110-164143-emb10-acc99'

    # Load model
    model = tf.keras.models.load_model(model_dir, compile=False)
    logging.info("loaded model from '%s'",model_dir)
    
    # Load node number to geoid encoding
    data = pd.read_csv('../data/detector/detector_surfaces.csv', dtype={'geo_id': np.uint64})
    
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
    reduced_dimension = 3
    logging.info("Start TSNE dimension reduction (%d -> %d)",embeddings.shape[1],reduced_dimension)
    embeddings_reduced = TSNE(n_components=reduced_dimension,n_jobs=20).fit_transform(embeddings)
    logging.info("Done, shape of reduced embeddings: %s",embeddings_reduced.shape)
    
    assert reduced_dimension == 2 or reduced_dimension == 3
    
    data['x'] = embeddings_reduced[:,0]
    data['y'] = embeddings_reduced[:,1]
        
    if reduced_dimension == 2:
        fig = px.scatter(data, x='x', y='y', color='volume', color_discrete_sequence=list(color_map.values()))
        fig.show()
    
    elif reduced_dimension == 3:
        data['z'] = embeddings_reduced[:,2]
        fig = px.scatter_3d(data, x='x', y='y', z='z', color='volume', color_discrete_sequence=list(color_map.values()))
        fig.show()
    
    
if __name__ == "__main__":
    main(sys.argv)
