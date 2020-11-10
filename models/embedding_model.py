import os
import sys
import datetime
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
from utility.data_import import *

def main(argv):
    logging.info("tensorflow devices: %s",[ t for n, t in tf.config.list_physical_devices() ])
    
    ######################
    # PREPARE GRAPH DATA #
    ######################
    
    nodes, geoid_edges, weights = generate_graph_from_data("../data/logger/data-201104-120629.csv")
    
    logging.info("num nodes: %s",len(nodes))
    logging.info("num edges: %s",len(geoid_edges))
    
    # get all valid nodes
    detector_data = pd.read_csv("../data/detector/detector_surfaces.csv", dtype={'geo_id': np.uint64})
    all_geo_ids = detector_data['geo_id'].to_numpy()
    all_numbers = detector_data['ordinal_id'].to_numpy()
    logging.info("detector coverage ratio: %s",len(nodes)/len(all_geo_ids))
    logging.info("surfaces missing: %s",len(all_geo_ids)-len(nodes))

    # Transform geoids to numbers starting from 0
    geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }

    connected_edges = np.empty(shape=geoid_edges.shape,dtype=np.int32)
    for number_edge, id_edge in zip(connected_edges,geoid_edges):
        number_edge[0] = geoid_to_number[id_edge[0]]
        number_edge[1] = geoid_to_number[id_edge[1]]
        
    edge_set = set([(e[0],e[1]) for e in connected_edges])

    ###########################
    # Training Data Generator #
    ###########################
    
    def generate_batch(batch_size):
        num_connected = batch_size//2
        num_unconnected = batch_size - num_connected
        
        
        while True:
            # connected edges
            idxs = np.arange(len(connected_edges))
            np.random.shuffle(idxs)
            
            x_connected = connected_edges[idxs[0:num_connected]].copy()
            y_connected = np.ones(num_connected)
            
            # unconnected edges
            x_unconnected = np.random.randint(0,len(all_geo_ids),(num_unconnected,2))
            
            for i in range(num_unconnected):
                while True:
                    new_sample = np.random.randint(0,len(all_geo_ids),2)
            
                    if new_sample[0] != new_sample[1] and (new_sample[0],new_sample[1]) not in this_edge_set:
                        x_unconnected[i] = new_sample
                        this_edge_set.add((new_sample[0],new_sample[1]))
                        break
                    
            y_unconnected = np.zeros(num_unconnected)
            
            # combine
            x = np.concatenate([x_connected,x_unconnected])
            y = np.concatenate([y_connected,y_unconnected])
            
            # shuffle
            assert len(x) == len(y)
            idxs = np.arange(len(x))
            np.random.shuffle(idxs)
            
            x = x[idxs]
            y = y[idxs]
            
            yield ([ x[:,0], x[:,1] ], y)

    ###################
    # LEARN EMBEDDING #
    ###################
    
    def my_accuracy(y_true, y_pred):
        num_correct = tf.math.reduce_sum(tf.cast(tf.math.equal(y_true, tf.math.round(y_pred)),tf.float32))
        num_total = tf.cast(tf.shape(y_true)[0],tf.float32)
        return tf.math.divide(num_correct,num_total)
    
    
    def build_model(num_categories,embedding_dim,learning_rate):
        input_node_1 = tf.keras.Input(1)
        input_node_2 = tf.keras.Input(1)
        
        embedding_layer = tf.keras.layers.Embedding(num_categories,embedding_dim)
        
        x = embedding_layer(input_node_1)
        y = embedding_layer(input_node_2)
        
        #z = tf.keras.layers.Dot(axes=2,normalize=True)([x,y])
        #z = tf.keras.layers.Reshape(target_shape=(1,))(z)
        
        x = tf.keras.layers.Reshape(target_shape=(embedding_dim,))(x)
        y = tf.keras.layers.Reshape(target_shape=(embedding_dim,))(y)
        z = tf.keras.layers.Concatenate()([x,y])
        z = tf.keras.layers.Dense(50)(z)
        z = tf.keras.layers.Activation(tf.nn.relu)(z)
        
        
        z = tf.keras.layers.Dense(1)(z)
        output = tf.keras.layers.Activation(tf.nn.sigmoid)(z)
        
        model = tf.keras.Model(inputs=[input_node_1,input_node_2],outputs=[output])
        
        model.summary()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.BinaryCrossentropy(),
            #metrics=[accuracy],
            metrics=[tf.keras.metrics.Accuracy(),my_accuracy]
        )
        
        return model

    model_params = {
        'num_categories': len(all_geo_ids),
        'embedding_dim': 10,
        'learning_rate': 0.01
    }
    
    logging.info("Start learning embedding")
    
    batch_size = 128
    gen = generate_batch(batch_size)
    
    train_embedding_model = build_model(**model_params)
    history = train_embedding_model.fit(
        x=gen, 
        steps_per_epoch=2*len(connected_edges)//batch_size,
        epochs=400,
        verbose=2
    )
    
    exit()
    
    # Embedding model
    input_node = tf.keras.Input(1)
    output_node = train_embedding_model.get_layer("embedding")(input_node)
    
    embedding_model = tf.keras.Model(input_node,output_node)
    
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_string = "-emb" + str(model_params['embedding_dim'])
    accuracy_string = "-acc" + str(round(history.history['accuracy'][-1]*100))
    output_filename = "../data/embeddings/" + date_string + embedding_string + accuracy_string
        
    embedding_model.save(output_filename)
    logging.info("exported model to '" + output_filename + "'")
    
    
if __name__ == "__main__":
    main(sys.argv)
