import os
import sys
import datetime
import argparse
import pprint
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

from timeit import Timer

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
from utility.data_import import *



def build_and_run_embedding_model(training_data_file, detector_data_file, output_dir, embedding_dim, learning_rate, batch_size, epochs):
    logging.info("tensorflow devices: %s",[ t for n, t in tf.config.list_physical_devices() ])
    
    ######################
    # PREPARE GRAPH DATA #
    ######################
        
    graph_data = pd.read_csv(training_data_file, dtype={'start_id': np.uint64, 'end_id': np.uint64})

    edges = graph_data[["start_id","end_id"]].to_numpy()
    
    nodes = np.unique(edges.flatten())

    # transform doublets in edges to weights
    geoid_edges, weights = np.unique(edges, axis=0, return_counts=True)
    
    logging.info("num nodes: %s",len(nodes))
    logging.info("num edges: %s",len(geoid_edges))
    
    # Get all valid nodes
    detector_data = pd.read_csv(detector_data_file, dtype={'geo_id': np.uint64})
    all_geo_ids = detector_data['geo_id'].to_numpy()
    all_numbers = detector_data['ordinal_id'].to_numpy()
    logging.info("detector coverage ratio: %s",len(nodes)/len(all_geo_ids))
    logging.info("surfaces missing: %s",len(all_geo_ids)-len(nodes))

    # Transformation dictionary
    geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }

    connected_edges = np.empty(shape=geoid_edges.shape,dtype=np.int32)
    for number_edge, id_edge in zip(connected_edges,geoid_edges):
        number_edge[0] = geoid_to_number[id_edge[0]]
        number_edge[1] = geoid_to_number[id_edge[1]]        
    
    # A bijective NxN -> N map, used for pair comparison
    def cantor_pairing(array):
        k0 = array[:,0]
        k1 = array[:,1]
        return 0.5 * (k0 + k1)*(k0 + k1 + 1) + k1
        
    def generate_batch(batch_size):
        num_connected = batch_size//2
        num_unconnected = batch_size - num_connected
        
        # increase number of unconnecteds to account for later filtered out edges
        increase = 1.1
        
        while True:
            # connected edges
            idxs = np.arange(len(connected_edges))
            np.random.shuffle(idxs)
            
            x_connected = connected_edges[idxs[0:num_connected]].copy()
            y_connected = np.ones(num_connected)
            
            # unconnected edges
            x_unconnected = np.random.randint(0,len(all_geo_ids),(int(num_unconnected*increase),2))
            x_unconnected = np.unique(x_unconnected,axis=0)
            
            # TODO apply unique before to speed up?
            is_unconnected_mask = np.logical_not(np.isin(cantor_pairing(x_unconnected),cantor_pairing(connected_edges),assume_unique=True))
            is_no_loop_mask = x_unconnected[:,0] != x_unconnected[:,1]
            valid_mask = np.logical_and(is_unconnected_mask, is_no_loop_mask)
            
            x_unconnected = x_unconnected[valid_mask]
            
            if len(x_unconnected) < num_unconnected:
                print("CONTINUE NECESSARY")
                continue
            
            x_unconnected = x_unconnected[0:num_unconnected]
            y_unconnected = np.zeros(num_unconnected)
            # combine
            x = np.concatenate([x_connected,x_unconnected])
            y = np.concatenate([y_connected,y_unconnected])
            
            # test properties
            #assert len(x) == len(np.unique(x,axis=0))
            #assert len(x) == len(y)
            
            # shuffle
            idxs = np.arange(len(x))
            np.random.shuffle(idxs)
            
            x = x[idxs]
            y = y[idxs]
            
            yield ([ x[:,0], x[:,1] ], y)

    ###################
    # LEARN EMBEDDING #
    ###################    
    
    def build_model(num_categories,embedding_dim,learning_rate):
        input_node_1 = tf.keras.Input(1)
        input_node_2 = tf.keras.Input(1)
        
        embedding_layer = tf.keras.layers.Embedding(num_categories,embedding_dim)
        
        x = embedding_layer(input_node_1)
        y = embedding_layer(input_node_2)
        
        z = tf.keras.layers.Dot(axes=2,normalize=True)([x,y])
        z = tf.keras.layers.Reshape(target_shape=(1,))(z)
            
        z = tf.keras.layers.Dense(1)(z)
        output = tf.keras.layers.Activation(tf.nn.sigmoid)(z)
        
        model = tf.keras.Model(inputs=[input_node_1,input_node_2],outputs=[output])
        
        model.summary()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy()]
        )
        
        return model

    model_params = {
        'num_categories': len(all_geo_ids),
        'embedding_dim': embedding_dim,
        'learning_rate': learning_rate
    }
    
    logging.info("Start learning embedding")
    
    gen = generate_batch(batch_size)
    
    train_embedding_model = build_model(**model_params)
    history = train_embedding_model.fit(
        x=gen, 
        steps_per_epoch=2*len(connected_edges)//batch_size,
        epochs=epochs,
        verbose=2
    )
    
    # Embedding model
    input_node = tf.keras.Input(1)
    output_node = train_embedding_model.get_layer("embedding")(input_node)
    
    embedding_model = tf.keras.Model(input_node,output_node)
    
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_string = "-emb" + str(model_params['embedding_dim'])
    accuracy_string = "-acc" + str(round(history.history['accuracy'][-1]*100))
    output_filename = date_string + embedding_string + accuracy_string
        
    embedding_model.save(output_dir + output_filename)
    logging.info("exported model to '" + output_dir + output_filename + "'")
    
    
if __name__ == "__main__":
    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    
    params = {
        'training_data_file': root_dir + "logger/embedding_training/data-201104-120629.csv",
        'detector_data_file': root_dir + "detector/detector_surfaces.csv",
        'output_dir': root_dir + "models/embeddings/",
        'embedding_dim': 50,
        'learning_rate': 0.01,
        'batch_size': 8192,
        'epochs': 20000
    }    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--embedding_dim", type=int, help="the dimension of the generated embedding")  
    parser.add_argument("-e", "--epochs", type=int, help="epochs the simulation is running")
    parser.add_argument("-l", "--learning_rate", type=float, help="the learning rate")
    
    args = parser.parse_args()
    
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    
    if args.embedding_dim:
        params['embedding_dim'] = args.embedding_dim
        
    if args.epochs:
        params['epochs'] = args.epochs
            
    assert params['embedding_dim'] > 0
    assert params['epochs'] > 0
    assert params['learning_rate'] > 0.0
    
    
    logging.info("run with the following params:")
    pprint.pprint(params, width=1)
    
    build_and_run_embedding_model(**params)
