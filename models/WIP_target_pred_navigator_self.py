#
# Concept of this model: 
# 
# * Train embedding directely inside model
# * [start_params + start_emb] -> target_emb
# * Feedforward network, optimized with MSE and regularizer to punish (start_emb - target_emb) = 0
#

import os
import sys
import argparse
import datetime
import logging
import time
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from common.preprocessing import *
from common.evaluation import *
from common.misc import *


##########################
# The feed-forward model #
##########################


@tf.keras.utils.register_keras_serializable(package='Custom', name='my_regularizer')
def my_regularizer(weight_matrix):
    '''
    This loss is used to panelize if the embedding layer maps everything to zero
    '''
    return 0.01 / tf.math.reduce_sum(tf.math.abs(weight_matrix))


def build_model(embedding_dim,num_categories,hidden_layers,activations,learning_rate):
    assert len(hidden_layers) >= 1
    assert len(activations) == len(hidden_layers)
    
    params = tf.keras.Input(4)
    start = tf.keras.Input(1)
    
    embedding_layer = tf.keras.layers.Embedding(num_categories,embedding_dim, embeddings_regularizer=my_regularizer)
    
    start_emb = embedding_layer(start)
    start_emb = tf.keras.layers.Reshape((embedding_dim,))(start_emb)
    
    a = tf.keras.layers.Concatenate()([params,start_emb])
    
    a = tf.keras.layers.Dense(hidden_layers[0],activation=activations[0])(a)
    
    for i in range(1,len(hidden_layers)):
        a = tf.keras.layers.Dense(hidden_layers[i],activation=activations[i])(a)
    
    output = tf.keras.layers.Dense(embedding_dim)(a)
    
    model = tf.keras.Model(inputs=[start,params],outputs=[output])
    
    return model



def fit(model, train_data, epochs, validation_split, batch_size, optimizer):
    '''
    Custom fit function for the 'target pred navigator II' model. 
    
    Parameters:
    * model (Keras model)
    * train_data (tuple): must (contain start_ids, params, end_ids)
    * epochs (int)
    * validation_split (float)
    * batch_size (int)
    * optimizer (Keras optimizer)
    
    Returns
    * dictionary: history object
    '''
    assert len(train_data) == 3
    
    # Validation split    
    train_start, val_start, train_params, val_params, train_target, val_target = \
        train_test_split(*train_data, test_size=validation_split)
    
    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_start, train_params, train_target))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_start, val_params, val_target))
    val_dataset = val_dataset.batch(batch_size)

    steps_per_epoch = len(train_start) // batch_size + 1
    
    times = []
    history = {
        'loss': [],
        'val_loss': []
    }
    
    loss_function = tf.keras.losses.MeanSquaredError()
    
    # The loop
    for epoch in range(epochs):        
        print("Epoch {}/{}".format(epoch,epochs))
        
        t0 = time.time()
        
        for start, params, target in train_dataset:
            with tf.GradientTape() as tape:
                y_true = tf.squeeze(model.get_layer("embedding")(target, training=True))
                y_pred = tf.squeeze(model([start, params], training=True))
                
                loss = loss_function(y_true, y_pred)
                
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            
            train_loss = loss
        
        for start, params, target in val_dataset:
            y_true = model.get_layer("embedding")(target)
            y_pred = model([start, params], training=True)
        
            val_loss = loss_function(y_true, y_pred)
            
        t1 = time.time()
        
        delta = t1 - t0
        times.append(delta)
        remaining = (epochs - epoch) * np.mean(np.array(times))
        remaining_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(remaining))
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print("{:.2f}s (~ {} remaining) - loss = {:.4f} - val_loss = {:.4f}".format(delta,
                                                                                    remaining_str,
                                                                                    train_loss,
                                                                                    val_loss),
              flush=True)
        
    return history



def evaluate_edge_graph(pos, x_emb, y_true, x_params, score_matrix, graph_edge_map, model):
    '''
    Evaluates one edge and puts result into a score_matrix. Evaluation is based on the graph.
    Intended to be used inside 'make_evaluation_plots()'.
    
    Parameters:
    * pos (int): position in track
    * x_emb (ndarray): start embedding
    * y_true (ndarray): true target embedding
    * x_params (ndarray): track parameters at start
    * score_matrix (dataframe): dataframe for results
    * nn (sklearn knn index)
    * model (Keras model)
    
    Returns:
    * dataframe: the modified score_matrix
    '''
    pass
    


#################
# Main function #
#################

def main():
    options = init_options_and_logger(get_navigation_training_dir(),
                                      get_root_dir() + "models/target_pred_navigator_self/")  

    ################
    # PREPARE DATA #
    ################
        
    prop_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64}) 
    
    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    # Beampipe split and new mapping
    prop_data = uniform_beampipe_split(prop_data, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data = geoid_to_ordinal_number(prop_data, detector_data, total_beampipe_split)
    
    # Build graph (needed for testing/scoring)
    graph_edge_map = generate_graph_edge_map(prop_data, total_node_num)
    
    # Categorize into tracks (also needed for testing later)
    selected_params = ['dir_x', 'dir_y', 'dir_z', 'qop']
    train_starts, train_params, train_targets = \
        categorize_into_tracks(prop_data, total_beampipe_split, selected_params)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(train_starts), max([ len(track) for track in train_starts ]))
    
    train_starts, test_starts, train_params, test_params, train_targets, test_targets = \
        train_test_split(train_starts, train_params, train_targets, test_size=options['test_split'])
    
    num_samples = len(np.concatenate(train_starts))
    y_dummys = np.zeros(num_samples)
    
    
    ###############################
    # Model building and training #
    ###############################
    
    # Build model
    model_params = {
        'embedding_dim': options['embedding_dim'],
        'num_categories': total_node_num,
        'hidden_layers': [ options['layer_size'] ] * options['network_depth'],
        'activations': [ options['activation'] ] * options['network_depth'],
        'learning_rate': options['learning_rate']
    }
    
    model = build_model(**model_params)
    
    # Do the training    
    logging.info("start training")
    
    history = fit(model, 
        (np.concatenate(train_starts).reshape(-1,1), 
         np.concatenate(train_params).reshape(-1,4), 
         np.concatenate(train_targets).reshape(-1,1)), 
        options['epochs'],
        options['validation_split'],
        options['batch_size'],
        tf.keras.optimizers.Adam(options['learning_rate'])
    )
        
    logging.info("finished training") 
    
    # Extract embedding model
    # TODO
     
    ##############
    # Evaluation #
    ##############
    
    # TODO use make_evaluation_plots(...)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.legend(['train_loss', 'val_loss'])
    
    if options['show']:
        plt.show()
    
if __name__ == "__main__":
    main()
