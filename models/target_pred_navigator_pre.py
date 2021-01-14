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
import json
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
        run_eagerly=True
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
    y_pred_dist = np.sum(np.square(y_pred - y_true))
    
    # all distances (sorted)
    start_id = int(np.squeeze(nn.kneighbors(x_emb.reshape(1,-1), 1, return_distance=False)))
    target_ids = graph_edge_map[ start_id ].targets
    
    target_embs = np.squeeze(emb_model(target_ids),axis=1)
    assert y_true in target_embs
    
    dists = np.sort(np.sum(np.square(target_embs - y_pred), axis=1))
    assert len(dists) == len(target_embs) == len(target_ids)
    
    # find score
    score = int(np.argwhere(np.equal(dists, y_pred_dist)))
    
    # Fill abs_score_matrix
    if score == 0: score_matrix.loc[pos, 'in1'] += 1
    elif score == 1: score_matrix.loc[pos, 'in2'] += 1
    elif score == 2: score_matrix.loc[pos, 'in3'] += 1
    elif score < 5: score_matrix.loc[pos, 'in5'] += 1
    elif score < 10: score_matrix.loc[pos, 'in10'] += 1
    else: score_matrix.loc[pos, 'other'] += 1
    
    # Fill res_scores (do the 1- to let the best result be 1)
    score_matrix.loc[pos, 'relative_score'] += 1 - score/len(target_embs)
    score_matrix.loc[pos, 'num_edges'] += len(target_embs)
    
    return score_matrix



#####################
# The main funciton #
#####################

def main(): 
    options = init_options_and_logger(get_navigation_training_dir(), 
                                      os.path.join(get_root_dir(), "models/target_pred_navigator/navigation/"),
                                      { 'evaluation_method': 'nn' })   
    
    assert options['evaluation_method'] == 'nn' or options['evaluation_method'] == 'graph'
    
    embedding_dir = os.path.join(get_root_dir(), 'models/target_pred_navigator/embeddings/')
    embedding_info = extract_embedding_model(embedding_dir, options['embedding_dim'])
    options['embedding_file'] = embedding_info.path
    options['beampipe_split_z'] = embedding_info.bpsplit_z
    options['beampipe_split_phi'] = embedding_info.bpsplit_phi
    
    logging.info("Imported embedding '%s'", options['embedding_file'])
    logging.info("Beampipe split set by embedding is (%d,%d)", options['beampipe_split_z'], options['beampipe_split_phi'])

    ########################
    # Import training data #
    ########################
    
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64})
    prop_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    
    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    #################
    # Preprocessing #
    #################    
    
    # Beampipe split and new mapping
    prop_data = beampipe_split(prop_data, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data = geoid_to_ordinal_number(prop_data, detector_data, total_beampipe_split)
    
    # Categorize into tracks (also needed for testing later)
    selected_params = ['dir_x', 'dir_y', 'dir_z', 'qop']
    x_track_ids, x_track_params, y_track_ids = \
        categorize_into_tracks(prop_data, total_beampipe_split, selected_params)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(x_track_ids), max([ len(track) for track in x_track_ids ]))
    
    # Train test split (includes shuffle)
    train_test_split_result = train_test_split(x_track_ids, x_track_params, y_track_ids, test_size=options['test_split'])
    
    # Don't need track structure for training
    x_train_ids = np.concatenate(train_test_split_result[0])
    x_train_params = np.concatenate(train_test_split_result[2])
    y_train_ids = np.concatenate(train_test_split_result[4])
    assert len(x_train_ids) == len(x_train_params) == len(y_train_ids)
    
    # Apply embedding
    embedding_model = tf.keras.models.load_model(options['embedding_file'], compile=False)
    assert options['embedding_dim'] == np.squeeze(embedding_model(0)).shape[0]
    
    x_train_embs = np.squeeze(embedding_model(x_train_ids))
    y_train_embs = np.squeeze(embedding_model(y_train_ids))
    logging.info("Applied pretrained embedding to training data")
    
    logging.debug("x_train_embs.shape: %s",x_train_embs.shape)
    logging.debug("x_train_pars.shape: %s",x_train_params.shape)
    logging.debug("y_train_embs.shape: %s",y_train_embs.shape)
    
    
    ###############################
    # Model building and Training #
    ###############################
    
    # Build model
    model_params = {
        'embedding_dim': options['embedding_dim'],
        'hidden_layers': [ options['layer_size'] ] * options['network_depth'],
        'activations': [ options['activation'] ] * options['network_depth'],
        'learning_rate': options['learning_rate']
    }
    
    model = build_feedforward_model(**model_params)
    
    # Do the training    
    logging.info("Start training")
    
    history = model.fit(
        x=[x_train_embs, x_train_params], 
        y=y_train_embs,
        validation_split=options['validation_split'],
        batch_size=options['batch_size'],
        epochs=options['epochs'],
        verbose=2,
        callbacks=[RemainingTimeEstimator(options['epochs'])],
    )
    
    logging.info("Finished training")  
    
    
    ##############
    # Evaluation #
    ##############
    
    # Apply embedding to test data
    x_test_embs = [ np.squeeze(embedding_model(ids)) for ids in train_test_split_result[1] ]
    x_test_params = train_test_split_result[3]
    y_test_embs = [ np.squeeze(embedding_model(ids)) for ids in train_test_split_result[5] ]
    assert len(x_test_embs) == len(x_test_params) == len(y_test_embs)
    logging.info("Applied embedding to test data")
    
    # Build nn index matching the embedding model
    nn = NearestNeighbors()
    nn.fit(np.squeeze(embedding_model(np.arange(total_node_num))))
    assert nn_index_matches_embedding_model(embedding_model, nn)
    
    graph_edge_map = generate_graph_edge_map(prop_data, total_node_num)
    
    # Do the evaluation
    if options['evaluation_method'] == 'nn':
        fig, axes, score = make_evaluation_plots(x_test_embs, y_test_embs, x_test_params, history.history,
                                                 lambda a,b,c,d,e: evaluate_edge_nn(a,b,c,d,e,nn,model))
    else: 
        fig, axes, score = make_evaluation_plots(x_test_embs, y_test_embs, x_test_params, history.history,
                                                 lambda a,b,c,d,e: evaluate_edge_graph(a,b,c,d,e,graph_edge_map,nn,
                                                                                       model,embedding_model))
    
    # Add additional information to figure
    bpsplit_str = "bp split: ({}, {})".format(options['beampipe_split_z'],options['beampipe_split_phi'])
    
    arch_str = "arch: [ "
    for size, activation in zip(model_params['hidden_layers'], model_params['activations']):
        arch_str += "({}, {}), ".format(size, activation)
    arch_str += "] "
    
    eval_str = "eval: {}".format(options['evaluation_method'])
    
    data_size_str = "n: {}".format(options['prop_data_size'])
    
    fig.suptitle("Target Predict I  -  {}  -  {}  -  {}  -  {}".format(data_size_str, bpsplit_str, arch_str, eval_str), fontweight='bold')
    
    if options['show']:
        plt.show()
       
    ##########   
    # Export #
    ##########
   
    eval_string = "-gr" if options['evaluation_method'] == 'graph' else "-nn"
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_string = "-emb{}".format(options['embedding_dim'])
    accuracy_string = "-acc{}".format(int(score*100))
    output_file = os.path.join(options['output_dir'], date_string + embedding_string + accuracy_string + eval_string)
    
    if options['export']: 
        model.save(output_file)
        logging.info("Exported model to '%s'",output_file)
    
        fig.savefig(output_file + '.png')
        logging.info("Exported chart to '%s.png'", output_file)
        
        with open(output_file + '.json','w') as conf_file:
            json.dump(options, conf_file, indent=4)
        logging.info("Exported configuration to '%s.json'", output_file)
        
    else:
        logging.info("Would export to '%s'", output_file)
    
    
if __name__ == "__main__":
    main()
