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

from common.preprocessing import *
from common.misc import *


def main():
    options = init_options_and_logger(get_embedding_training_dir(),
                                      os.path.join(get_root_dir(), "models/target_pred_navigator/embeddings/"),
                                      { 'prop_data_size': 128 })
    
    ######################
    # PREPARE GRAPH DATA #
    ######################
        
    graph_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64})

    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split

    graph_data = beampipe_split(graph_data, options['beampipe_split_z'], options['beampipe_split_phi'])        
    graph_data = geoid_to_ordinal_number(graph_data, detector_data, total_beampipe_split)
    
    connected_edges = np.unique(graph_data[['start_id', 'end_id']].to_numpy(), axis=0)
    logging.info("Imported graph with %d edges", len(connected_edges))
       
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
            x_unconnected = np.random.randint(0,total_node_num,(int(num_unconnected*increase),2))
            x_unconnected = np.unique(x_unconnected,axis=0)
            
            is_unconnected_mask = np.logical_not(np.isin(cantor_pairing(x_unconnected),cantor_pairing(connected_edges),assume_unique=True))
            is_no_loop_mask = x_unconnected[:,0] != x_unconnected[:,1]
            valid_mask = np.logical_and(is_unconnected_mask, is_no_loop_mask)
            
            x_unconnected = x_unconnected[valid_mask]
            
            if len(x_unconnected) < num_unconnected:
                logging.warning("'continue' in generator loop")
                continue
            
            x_unconnected = x_unconnected[0:num_unconnected]
            y_unconnected = np.zeros(num_unconnected)
            
            # combine
            x = np.concatenate([x_connected,x_unconnected])
            y = np.concatenate([y_connected,y_unconnected])
            
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
        
        #model.summary()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy()]
        )
        
        return model

    model_params = {
        'num_categories': total_node_num,
        'embedding_dim': options['embedding_dim'],
        'learning_rate': options['learning_rate']
    }
    
    gen = generate_batch(options['batch_size'])
    
    train_embedding_model = build_model(**model_params)
    
    steps_per_epoch = 2*len(connected_edges)//options['batch_size']
    logging.info("Start to fit embedding (steps_per_epoch = %d)", steps_per_epoch)
    
    history = train_embedding_model.fit(
        x=gen, 
        steps_per_epoch=2*len(connected_edges)//options['batch_size'],
        epochs=options['epochs'],
        verbose=2,
        callbacks=[RemainingTimeEstimator(options['epochs'])]
    )
    
    logging.info("Finished fitting");
    
    fig, axes = plt.subplots(1,3, figsize=(16,6))
    
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].plot(history.history['loss'])
    
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].plot(history.history['accuracy'])
    
    axes[2].set_xlabel("epochs")
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("Binary accuracy")
    axes[2].plot(history.history['binary_accuracy'])
    
    fig.suptitle("Embedding training: batch_size={}".format(options['batch_size']), fontweight='bold')
    
    fig.text(0,0,"Training data: " + options['propagation_file'])
    
    if options['show']:
        plt.show()
    
    ##########
    # Export #
    ##########
    
    input_node = tf.keras.Input(1)
    output_node = train_embedding_model.get_layer("embedding")(input_node)
    
    embedding_model = tf.keras.Model(input_node,output_node)
    
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_string = "-emb{}".format(model_params['embedding_dim'])
    accuracy_string = "-acc{}".format(round(history.history['accuracy'][-1]*100))
    bpsplit_z_string = "-bz{}".format(options['beampipe_split_z'])
    bpsplit_phi_string = "-bp{}".format(options['beampipe_split_phi'])
    output_filename = date_string + embedding_string + accuracy_string + bpsplit_z_string + bpsplit_phi_string
            
    if options['export']:
        embedding_model.save(options['output_dir'] + output_filename)
        logging.info("exported model to '" + options['output_dir'] + output_filename + "'")
        
        fig.savefig(options['output_dir'] + output_filename + ".png")
        logging.info("exported figure to '" + options['output_dir'] + output_filename + ".png'")
    else:
        logging.info("would export model to '" + options['output_dir'] + output_filename + "'")
        
    
    
if __name__ == "__main__":
    main()