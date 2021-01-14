#
# Concept of this model: 
# 
# * Use pretrained embedding layer
# * [start_params + start_emb] -> target_emb
# * Feedforward network, optimized with MSE
# 

import os
import sys
import argparse

import datetime
import logging
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from common.misc import *
from common.preprocessing import *
from common.evaluation import *

##########################
# The feed-forward model #
##########################

def build_feedforward_model(embedding_dim,hidden_layers,activations,learning_rate):
    '''
    Function that builds and compiles the model. It is of form:
    
    [start_embedding, start_params] -> [target_embedding]
    
    Parameters:
    * embedding_dim (int): dim of the pre-trained embedding
    * hidden_layers (list): list of layer sizes
    * activations (list): list of activations
    * learning_rate (float)
    
    Returns:
    * compiled Keras model
    '''
    assert len(hidden_layers) >= 1
    assert len(activations) == len(hidden_layers)
    
    input_params = tf.keras.Input(4)
    input_embs = tf.keras.Input(embedding_dim)
    
    a = tf.keras.layers.Concatenate()([input_params,input_embs])
    
    a = tf.keras.layers.Dense(hidden_layers[0],activation=activations[0])(a)
    
    for i in range(1,len(hidden_layers)):
        a = tf.keras.layers.Dense(hidden_layers[i],activation=activations[i])(a)
    
    output_embs = tf.keras.layers.Dense(embedding_dim)(a)
    
    model = tf.keras.Model(inputs=[input_embs,input_params],outputs=[output_embs])
    #model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=tf.keras.losses.MeanSquaredError(),
        #run_eagerly=True
    )
    
    return model



def evaluate_edge_nn(pos, x_emb, y_true, x_params, score_matrix, nn, model):
    '''
    Evaluates one edge and puts result into a score_matrix. Evaluation is based on a nn index.
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
    y_pred = model([x_emb.reshape(1,-1), x_params.reshape(1,-1)])
    
    y_true_idx = int(np.squeeze(nn.kneighbors(y_true.reshape(1,-1), 1, return_distance=False)))
    y_pred_idxs = np.squeeze(nn.kneighbors(y_pred, 10, return_distance=False))
    
    if y_pred_idxs[0] == y_true_idx: 
        score_matrix.loc[pos, 'in1'] += 1
    elif y_pred_idxs[1] == y_true_idx: 
        score_matrix.loc[pos, 'in2'] += 1
    elif y_pred_idxs[2] == y_true_idx: 
        score_matrix.loc[pos, 'in3'] += 1
    elif np.equal(y_pred_idxs[3:5], y_true_idx).any(): 
        score_matrix.loc[pos, 'in5'] += 1
    elif np.equal(y_pred_idxs[5:], y_true_idx).any(): 
        score_matrix.loc[pos, 'in10'] += 1
    else: 
        score_matrix.loc[pos, 'other'] += 1
    
    return score_matrix



def evaluate_edge_graph(pos, x_emb, y_true, x_params, score_matrix, graph_edge_map, nn, nav_model, emb_model):
    # predicted distance 
    y_pred = np.squeeze(nav_model([x_emb.reshape(1,-1), x_params.reshape(1,-1)]))
    pred_dist = np.sum(np.square(y_pred - y_true))
    
    # all distances (sorted)
    start_id = int(np.squeeze(nn.kneighbors(x_emb.reshape(1,-1), 1, return_distance=False)))
    target_ids = graph_edge_map[ start_id ].targets
    
    targets = np.squeeze(emb_model(target_ids),axis=1)
    assert y_true in targets
    
    dists = np.sort(np.sum(np.square(targets - y_pred), axis=1))
    assert len(dists) == len(targets) == len(target_ids)
    
    # find score
    score = int(np.argwhere(np.equal(dists, pred_dist)))
    
    # Fill abs_score_matrix
    if score == 0: score_matrix.loc[pos, 'in1'] += 1
    elif score == 1: score_matrix.loc[pos, 'in2'] += 1
    elif score == 2: score_matrix.loc[pos, 'in3'] += 1
    elif score < 5: score_matrix.loc[pos, 'in5'] += 1
    elif score < 10: score_matrix.loc[pos, 'in10'] += 1
    else: score_matrix.loc[pos, 'other'] += 1
    
    # Fill res_scores (do the 1- to let the best result be 1)
    score_matrix.loc[pos, 'relative_score'] += 1 - score/len(targets)
    score_matrix.loc[pos, 'num_edges'] += len(targets)
    
    return score_matrix



#####################
# The main funciton #
#####################

def main(): 
    options = init_options_and_logger(get_navigation_training_dir(),
                                      os.path.join(get_root_dir(), "models/target_pred_navigator/navigation/"),
                                      { 'evaluation_method': 'graph' })   
    
    assert options['evaluation_method'] == 'nn' or options['evaluation_method'] == 'graph'
    
    embedding_dir = os.path.join(get_root_dir(), 'models/target_pred_navigator/embeddings/')
    embedding_info = extract_embedding_model(embedding_dir, options['embedding_dim'])
    options['embedding_file'] = embedding_info.path
    options['beampipe_split_z'] = embedding_info.bpsplit_z
    options['beampipe_split_phi'] = embedding_info.bpsplit_phi

    ########################
    # Import training data #
    ########################
    
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64})
    prop_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    
    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    # Beampipe split and new mapping
    prop_data = beampipe_split(prop_data, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data = geoid_to_ordinal_number(prop_data, detector_data, total_beampipe_split)
    
    # Categorize into tracks (also needed for testing later)
    selected_params = ['dir_x', 'dir_y', 'dir_z', 'qop']
    train_tracks_start, train_tracks_params, train_tracks_end = \
        categorize_into_tracks(prop_data, total_beampipe_split, selected_params)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(train_tracks_start), max([ len(track) for track in train_tracks_start ]))
    
    # Load embedding
    embedding_model = tf.keras.models.load_model(options['embedding_file'], compile=False)
    embedding_dim = np.squeeze(embedding_model(0)).shape[0]
    
    # Apply embedding
    for i in range(len(train_tracks_start)):
        train_tracks_start[i] = np.squeeze(embedding_model(train_tracks_start[i]))
        train_tracks_end[i] = np.squeeze(embedding_model(train_tracks_end[i]))
    logging.info("Applied pretrained embedding")
    
    # Train test split (includes shuffle)
    x_train_embs, x_test_embs, x_train_pars, x_test_pars, y_train_embs, y_test_embs = \
        train_test_split(train_tracks_start, train_tracks_params, train_tracks_end, test_size=options['test_split'])
    
    # don't need track structure here
    x_train_embs = np.concatenate(x_train_embs)
    x_train_pars = np.concatenate(x_train_pars)
    y_train_embs = np.concatenate(y_train_embs)
    
    logging.debug("x_train_embs.shape: %s",x_train_embs.shape)
    logging.debug("x_train_pars.shape: %s",x_train_pars.shape)
    logging.debug("y_train_embs.shape: %s",y_train_embs.shape)
    
    ###############################
    # Model building and Training #
    ###############################
    
    # Build model
    model_params = {
        'embedding_dim': embedding_dim,
        'hidden_layers': [ options['layer_size'] ] * options['network_depth'],
        'activations': [ options['activation'] ] * options['network_depth'],
        'learning_rate': options['learning_rate']
    }
    
    model = build_feedforward_model(**model_params)
    
    # Do the training    
    logging.info("start training")
    
    history = model.fit(
        x=[x_train_embs, x_train_pars], 
        y=y_train_embs,
        validation_split=options['validation_split'],
        batch_size=options['batch_size'],
        epochs=options['epochs'],
        verbose=2
    )
    
    logging.info("finished training")  
    
    
    ##############
    # Evaluation #
    ##############
    
    # Build nn index
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64})
    all_embeddings = np.squeeze(embedding_model(detector_data["ordinal_id"].to_numpy()))
    
    nn = NearestNeighbors()
    nn.fit(all_embeddings)
    
    # Do the evaluation
    
    if options['evaluation_method'] == 'nn':
        fig, axes, score = make_evaluation_plots(x_test_embs, y_test_embs, x_test_pars, history.history,
                                                 lambda a,b,c,d,e: evaluate_edge_nn(a,b,c,d,e,nn,model))
    else: 
        graph_edge_map = generate_graph_edge_map(prop_data, total_node_num)
        
        fig, axes, score = make_evaluation_plots(x_test_embs, y_test_embs, x_test_pars, history.history,
                                                 lambda a,b,c,d,e: evaluate_edge_graph(a,b,c,d,e,graph_edge_map,nn,
                                                                                       model,embedding_model))
    
    # Add additional information to figure
    bpsplit_str = "bp split: z={}, phi={}".format(options['beampipe_split_z'],options['beampipe_split_phi'])
    
    arch_str = "arch: [ "
    for size, activation in zip(model_params['hidden_layers'], model_params['activations']):
        arch_str += "({}, {}), ".format(size, activation)
    arch_str += "] "
    
    lr_str = "lr: {} ".format(options['learning_rate'])
    
    eval_str = "eval: {}".format(options['evaluation_method'])
    
    fig.suptitle("NN based forward: {} - {} - {} - {}".format(bpsplit_str, arch_str, lr_str, eval_str), fontweight='bold')
    fig.text(0,0,"Training data: " + options['propagation_file'])
    
    if options['show']:
        plt.show()
       
    ##########   
    # Export #
    ##########
   
    if options['export']: 
        method_string = "-forward"
        date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        embedding_string = "-emb" + str(embedding_dim)
        accuracy_string = "-acc" + str(int(score*100))
        output_filename = date_string + method_string + embedding_string + accuracy_string
    
        model.save(options['output_dir'] + output_filename)
        logging.info("exported model to '" + options['output_dir'] + output_filename + "'")
    
        fig.savefig(options['output_dir'] + output_filename + ".png")
        logging.info("exported chart to '" + options['output_dir'] + output_filename + ".png" + "'")
    
    
if __name__ == "__main__":
    main()
