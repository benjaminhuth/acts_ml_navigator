import os
import sys
import datetime
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_addons as tfa
tf.get_logger().setLevel('ERROR')

from sklearn.neighbors import NearestNeighbors

from utility.data_import import *

def main(argv):
    # Load best model
    embedding_prefix = "../data/embeddings/"
    model_dirs = get_sorted_model_dirs(embedding_prefix)
    embedding_encoder = tf.keras.models.load_model(embedding_prefix + model_dirs[0], compile=False)
    logging.info("loaded model from '" + embedding_prefix + model_dirs[0] + "'")
    
    embedding_shape = embedding_encoder(0).shape
    logging.info("embedding shape: %s",str(embedding_shape))
    
    # Load ordinal numbers to geoid dictionary
    all_geoids = get_list_of_all_geoids("../data/event000000000-detectors.csv")
    all_numbers = np.arange(len(all_geoids))
    geoid_to_numbers = { geoid: number for geoid,number in zip(all_geoids,all_numbers) }
    
    # Establish neighbouring index
    all_embeddings = np.squeeze(embedding_encoder(all_numbers).numpy())
    nn = NearestNeighbors()
    nn.fit(all_embeddings)
    logging.info("built neighbouring index")
    
    # Custom accuracy function for neigbhors
    def has_correct_nearest_neighbour(y_true, y_pred):
        idxs_a = nn.kneighbors(y_true.numpy(), 1, return_distance=False)
        idxs_b = nn.kneighbors(y_pred.numpy(), 1, return_distance=False)
        
        return np.sum(idxs_a == idxs_b)/len(idxs_a)
        
    def neighbor_accuracy(y_true, y_pred):
        return tf.py_function(func=has_correct_nearest_neighbour, inp=[y_true,y_pred], Tout=tf.float32)
    
    ##########################
    # Generate training data #
    ##########################
    
    # Import data
    data_prefix = "../data/logger/"
    data_file = os.listdir(data_prefix)[0]
    nodes, edges, weights, dirs = generate_graph_from_data(data_prefix + data_file, export_dir=True)
    logging.info("loaded data from '" + data_prefix + data_file + "'")
    
    # Extract data
    x_train_ids = edges[:,0]
    x_train_dirs = dirs
    y_train = edges[:,1]
    
    # Transform data
    x_train_ids = np.array([ geoid_to_numbers[geoid] for geoid in x_train_ids ])
    x_train_ids = embedding_encoder(x_train_ids)
    x_train_ids = np.reshape(x_train_ids,newshape=(x_train_ids.shape[0],x_train_ids.shape[2]))
    
    y_train = np.array([ geoid_to_numbers[geoid] for geoid in y_train ])
    y_train = embedding_encoder(y_train)
    y_train = np.reshape(y_train,newshape=(y_train.shape[0],y_train.shape[2]))
    
    logging.info("built training data")    
    logging.debug("x_train_ids.shape: %s",str(x_train_ids.shape))
    logging.debug("x_train_dirs.shape: %s",str(x_train_dirs.shape))
    logging.debug("y_train.shape: %s",str(y_train.shape))    
    
    def build_navigation_model(embedding_dim,hidden_layers,learning_rate):
        assert len(hidden_layers) >= 1
        
        input_id = tf.keras.Input(embedding_dim)
        input_dir = tf.keras.Input(3)
        
        x = tf.keras.layers.Dense(embedding_dim)(input_dir)
        
        x = tf.keras.layers.Concatenate()([input_id,x])
        
        x = tf.keras.layers.Dense(hidden_layers[0])(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        
        for i in range(1,len(hidden_layers)):
            x = tf.keras.layers.Dense(hidden_layers[i])(x)
            x = tf.keras.layers.Activation(tf.nn.relu)(x)
            
        output_id = tf.keras.layers.Dense(embedding_dim)(x)
        
        model = tf.keras.Model(inputs=[input_id,input_dir],outputs=[output_id])
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[neighbor_accuracy]
        )
        
        return model
    
    model_params = {
        'embedding_dim': embedding_shape[0],
        'hidden_layers': [100,100],
        'learning_rate': 0.001
    }
    
    navigation_model = build_navigation_model(**model_params)
    
    logging.info("start training")
    navigation_model.fit(
        x=[ x_train_ids, x_train_dirs ], 
        y=y_train,
        validation_split=0.33,
        batch_size=32,
        epochs=30,
        verbose=2
    )
        
    
    
if __name__ == "__main__":
    main(sys.argv)
