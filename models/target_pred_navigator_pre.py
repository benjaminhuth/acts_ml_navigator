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
import kerastuner as kt

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from common.misc import *
from common.preprocessing import *
from common.evaluation import *
from common.real_space_embedding import *

##########################
# The feed-forward model #
##########################

def build_feedforward_model(embedding_dim, hidden_layers, activations, learning_rate):
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
    
    model = tf.keras.Model(inputs=[input_embs,input_params], outputs=[output_embs], name="feedforward")
    #model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=tf.keras.losses.MeanSquaredError(),
        #run_eagerly=True
    )
    
    return model



def build_recurrent_model(embedding_dim, hidden_layers, activations, learning_rate, masking_value):
    '''
    Function that builds and compiles a recurrent model. It should return the whole output serias and contain a masking layer.
    
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
    
    input_params = tf.keras.Input(shape=(None,4))
    input_embs = tf.keras.Input(shape=(None,embedding_dim))
    
    a = tf.keras.layers.Concatenate()([input_params,input_embs])
    a = tf.keras.layers.Masking(mask_value=masking_value)(a)
    
    a = tf.keras.layers.GRU(hidden_layers[0],activation='tanh',recurrent_activation='sigmoid',return_sequences=True)(a)
    
    for i in range(1,len(hidden_layers)):
        a = tf.keras.layers.GRU(hidden_layers[i],activation='tanh',recurrent_activation='sigmoid',return_sequences=True)(a)
    
    output_embs = tf.keras.layers.Dense(embedding_dim)(a)
    
    model = tf.keras.Model(inputs=[input_embs,input_params], outputs=[output_embs], name="recurrent")
    
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=tf.keras.losses.MeanSquaredError(),
        #run_eagerly=True
    )
    
    return model
    



def hyperparameter_optimization(x_train_embs, x_train_params, y_train_embs, model_fn, options, masking_value=None):
    '''
    Perform hyperparameter optimization with keras tuner
    
    Parameters:
    * x_train_embs: embeddings
    * x_train_params: track parameters
    * y_train_embs: true result embeddings
    * model_fn: callable which builds the model
    * options: options dictionary
    * [OPT] masking_value: how are padded values masked (just for recurrent networks)
    '''
    logging.info("Performing Hyperband Optimization on model. After this was done, the result is printed, and we exit without exporting any results")
    
    def build_model_hpopt(hp, embedding_dim):
        layers = [ hp.Int('layer_size', 100, 1000) ] * hp.Int('network_depth', 2, 7)
        activations = [ 'relu' ] * len(layers)
            
        model_params = {
            'embedding_dim': embedding_dim,
            'hidden_layers': layers,
            'activations': activations,
            'learning_rate': hp.Choice('learning_rate', [0.01, 0.001, 0.0001]),
        }
        
        if masking_value != None:
            model_params['masking_value'] = masking_value
        
        return model_fn(**model_params)
    
    tuner = kt.tuners.Hyperband(
        lambda hp: build_model_hpopt(hp, options['embedding_dim']),
        objective='val_loss',
        max_epochs=options['epochs'],
        executions_per_trial=3,
        factor=3,
        directory="/home/benjamin/",
        project_name="{}-target-pred-hyperband".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    
    tuner.search(
        x=[x_train_embs, x_train_params],
        y=y_train_embs, 
        epochs=options['epochs'], 
        validation_split=options['validation_split'],
        batch_size=options['batch_size'],
        verbose=2,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        ]
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    logging.info("TUNING RESULTS:")
    logging.info("layer size:",best_hps.get('layer_size'))
    logging.info("network depth:", best_hps.get('network_depth'))
    logging.info("learning rate:", best_hps.get('learning_rate'))




def evaluate_edge_nn(pos, x_emb, y_true, x_params, result, nn, model, graph_map):
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
    * graph_map (dictionary): { id : targets, weights, position }
    
    Returns:
    * dataframe: the modified score_matrix
    '''
    if model.name == "recurrent" and pos == 0:
        model.reset_states()
        
    # Input shapes differ if it is recurrent
    if model.name == "recurrent":
        y_pred = np.squeeze(model([x_emb.reshape(1,1,-1), x_params[3:7].reshape(1,1,-1)]))
    else:
        y_pred = np.squeeze(model([x_emb.reshape(1,-1), x_params[3:7].reshape(1,-1)]))
        
    x_idx = int(np.squeeze(nn.kneighbors(x_emb.reshape(1,-1), 1, return_distance=False)))
    y_true_idx = int(np.squeeze(nn.kneighbors(y_true.reshape(1,-1), 1, return_distance=False)))
    y_pred_idxs = np.squeeze(nn.kneighbors(y_pred.reshape(1,-1), 10, return_distance=False))
    
    score = np.argwhere(np.equal(y_pred_idxs, y_true_idx))
    
    # Here it can happen that it is not found, so we need workaround
    if len(score) == 0:
        score = 11 # so it gets in category 'other'
    else:
        score = int(score)
    
    return fill_in_results(pos, score, graph_map[ x_idx ].position[2], x_params, result)



def evaluate_edge_graph(pos, x_emb, y_true, x_params, result, graph_map, nn, nav_model, emb_model):
    '''
    Evaluation function for graph evaluation. It scores based on the distance between the prediction embedding and the true result embedding, where the candidates are not the whole space but only what is in the graph. Thus we need additionally the embedding model, to compute the embedding from the graph entries (which are just numeric IDs).
    
    Parameters:
    * pos (int): position in track
    * x_emb (ndarray): start embedding
    * y_true (ndarray): true target embedding
    * x_params (ndarray): track parameters at start
    * result (dataframe): dataframe for results
    * graph_map (dictionary): { id : targets, weights, position }
    * nn (sklearn knn index)
    * nav_model (Keras model): Model used for target prediction
    * emb_model (Keras model): Model used for embedding computation
    
    Returns:
    * dataframe: the modified score_matrix
    '''
    if pos == 0:
        nav_model.reset_states()
        
    # predicted distance 
    y_pred = np.squeeze(nav_model([x_emb.reshape(1,-1), x_params[3:7].reshape(1,-1)]))
    y_pred_dist = np.sum(np.square(y_pred - y_true))
    
    # all distances (sorted)
    start_id = int(np.squeeze(nn.kneighbors(x_emb.reshape(1,-1), 1, return_distance=False)))
    node_data = graph_map[ start_id ]
    
    target_ids = node_data.targets
    target_embs = np.squeeze(emb_model(target_ids),axis=1)
    assert y_true in target_embs
    
    dists = np.sort(np.sum(np.square(target_embs - y_pred), axis=1))
    assert len(dists) == len(target_embs) == len(target_ids)
    
    # find score    
    score = int(np.argwhere(np.equal(dists, y_pred_dist)))
    
    # Fill abs_score_matrix
    return fill_in_results(pos, score, node_data.position[2], x_params, result)



#####################
# The main funciton #
#####################

def main(): 
    options = init_options_and_logger(get_navigation_training_dir(), 
                                      os.path.join(get_root_dir(), "models/target_pred_navigator_pre/"),
                                      {
                                          'evaluation_method': 'nn',
                                          'hyperparameter_optimization': False,
                                          'concat_real_space_and_emb': False,
                                          'normalize_real_space_emb': True,
                                          'recurrent': False,
                                      })   
    
    assert options['evaluation_method'] == 'nn' or options['evaluation_method'] == 'graph'
    assert not (options['use_real_space_as_embedding'] and options['concat_real_space_and_emb'])
    
    embedding_dir = os.path.join(get_root_dir(), 'models/embeddings/')
    embedding_info = extract_embedding_model(embedding_dir, options['embedding_dim'], options['bpsplit_z'],
                                             options['bpsplit_phi'], options['bpsplit_method'])
    options['embedding_file'] = embedding_info.path
    options['bpsplit_z'] = embedding_info.bpsplit_z
    options['bpsplit_phi'] = embedding_info.bpsplit_phi
    
    logging.info("Imported embedding '%s'", options['embedding_file'])
    logging.info("Beampipe split set by embedding is (%d,%d)", (len(options['bpsplit_z'])-1), options['bpsplit_phi'])
    
    if options['use_real_space_as_embedding']:
        logging.warning("Imported embedding will NOT be used, instead a real-space embedding with the imported beampipe split")
    elif options['concat_real_space_and_emb']:
        logging.warning("The surface center position will be concatenated with the embedding")
        

    ########################
    # Import training data #
    ########################
    
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64})
    prop_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    
    total_beampipe_split = (len(options['bpsplit_z'])-1)*options['bpsplit_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    #################
    # Preprocessing #
    #################    
    
    # Beampipe split and new mapping
    prop_data, z_dist = apply_beampipe_split(prop_data, options['bpsplit_z'], options['bpsplit_phi'], return_z_distribution=True)
    prop_data = geoid_to_ordinal_number(prop_data, detector_data, total_beampipe_split)
    
    # Plot and save z bin distribution
    #plt.plot(z_dist[0], z_dist[1], ".-")
    #plt.xlabel("z-bins")
    #plt.ylabel("# of hits on z-bin")
    #plt.title("Hits per z-bin (z-split = {}, n={})".format(len(options['bpsplit_z'])-1, options['prop_data_size']))
    #plt.show()
    #exit()
    
    # Categorize into tracks (also needed for testing later)
    selected_params = ['pos_x', 'pos_y', 'pos_z', 'dir_x', 'dir_y', 'dir_z', 'qop']
    x_track_ids, x_track_params, y_track_ids = \
        categorize_into_tracks(prop_data, total_beampipe_split, selected_params)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(x_track_ids), max([ len(track) for track in x_track_ids ]))
    
    # Train test split (includes shuffle)
    train_test_split_result = train_test_split(x_track_ids, x_track_params, y_track_ids, test_size=options['test_split'])
    
    # Load embedding
    if options['use_real_space_as_embedding']:
        embedding_model = make_real_space_embedding_model(detector_data, options['bpsplit_z'], options['bpsplit_phi'],
                                                          normalize=options['normalize_real_space_emb'])
        options['embedding_dim'] = 3
    elif options['concat_real_space_and_emb']:
        real_space_model = make_real_space_embedding_model(detector_data, options['bpsplit_z'], options['bpsplit_phi'],
                                                           normalize=options['normalize_real_space_emb'])
        trained_emb_model = tf.keras.models.load_model(options['embedding_file'], compile=False)
        
        embedding_model = lambda i: np.expand_dims(np.hstack([ np.squeeze(real_space_model(i)), np.squeeze(trained_emb_model(i)) ]), 1)
        options['embedding_dim'] = options['embedding_dim'] + 3
    else:
        embedding_model = tf.keras.models.load_model(options['embedding_file'], compile=False)
        
    # Ensure dimensionality is correct (use list, since pseudo-models (lambdas) only process iterables
    assert options['embedding_dim'] == np.squeeze(embedding_model([0])).shape[0]
    logging.info("Loaded embedding with dim %d", options['embedding_dim'])
        
    # Apply embedding
    progress_bar = ProgressBar(len(train_test_split_result[0]), prefix="Applying embedding:")
    
    for i in range(len(train_test_split_result[0])):
        train_test_split_result[0][i] = np.squeeze(embedding_model(train_test_split_result[0][i]))
        train_test_split_result[4][i] = np.squeeze(embedding_model(train_test_split_result[4][i]))
        
        progress_bar.print_bar()
    
    logging.info("Applied pretrained embedding to training data")
        
    
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
    
    # Feedforward network
    if options['recurrent'] == False:
        logging.info("Do feed-forward training")
        
        # Don't need track structure for feed forward training
        x_train_embs = np.concatenate(train_test_split_result[0])
        x_train_params = np.concatenate(train_test_split_result[2])
        y_train_embs = np.concatenate(train_test_split_result[4])
        assert len(x_train_embs) == len(x_train_params) == len(y_train_embs)
        
        logging.debug("x_train_embs.shape: %s",x_train_embs.shape)
        logging.debug("x_train_pars.shape: %s",x_train_params.shape)
        logging.debug("y_train_embs.shape: %s",y_train_embs.shape)
        
        if options['hyperparameter_optimization']:
            hyperparameter_optimization(x_train_embs, x_train_params[:,3:7], y_train_embs, 
                                        build_feedforward_model, options)
            exit()
        
        model = build_feedforward_model(**model_params)
        
    # recurrent network
    else:
        logging.info("Do recurrent training")
        
        # Do padding for recurrent network
        padding_config = {
            'dtype': np.float32,
            'padding': 'post',
            'value': 0.0
        }
        
        x_train_embs = tf.keras.preprocessing.sequence.pad_sequences(train_test_split_result[0], **padding_config)
        x_train_params = tf.keras.preprocessing.sequence.pad_sequences(train_test_split_result[2], **padding_config)
        y_train_embs = tf.keras.preprocessing.sequence.pad_sequences(train_test_split_result[4], **padding_config)
        
        logging.debug("x_train_embs.shape: %s",x_train_embs.shape)
        logging.debug("x_train_pars.shape: %s",x_train_params.shape)
        logging.debug("y_train_embs.shape: %s",y_train_embs.shape)
        
        model_params['masking_value'] = padding_config['value']
        
        if options['hyperparameter_optimization']:
            hyperparameter_optimization(x_train_embs, x_train_params[:,3:7], y_train_embs, 
                                        build_recurrent_model, options, padding_config['value'])
            exit()
        
        model = build_recurrent_model(**model_params)
        
        
    history = model.fit(
        x=[x_train_embs, x_train_params[:,3:7]], 
        y=y_train_embs,
        validation_split=options['validation_split'],
        batch_size=options['batch_size'],
        epochs=options['epochs'],
        verbose=2,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            RemainingTimeEstimator(options['epochs']),
        ],
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
    is_keras_model = not (options['use_real_space_as_embedding'] or options['concat_real_space_and_emb'])
    assert nn_index_matches_embedding_model(embedding_model, nn, is_keras_model=is_keras_model)
    
    graph_map = make_graph_map(prop_data, total_node_num)
    
    # Do the evaluation
    evaluation_params = {
        'tracks_edges_start': x_test_embs,
        'tracks_params': x_test_params,
        'tracks_edges_target': y_test_embs,
        'history': history.history,
        'smooth_rzmap': options['eval_smooth_rzmap'],
        'smooth_radius': options['eval_smooth_rzmap_radius'],
    }
    
    if options['evaluation_method'] == 'nn':
        evaluation_params['evaluate_edge_fn'] = lambda a,b,c,d,e: evaluate_edge_nn(a,b,c,d,e,nn,model,graph_map)
        fig, axes, score = evaluate_and_plot(**evaluation_params)
    else: 
        evaluation_params['evaluate_edge_fn'] = lambda a,b,c,d,e: evaluate_edge_graph(a,b,c,d,e,graph_map,nn,model,embedding_model)
        fig, axes, score = evaluate_and_plot(**evaluation_params)
    
    # Add additional information to figure
    bpsplit_str = "bp split: ({}, {})".format((len(options['bpsplit_z'])-1),options['bpsplit_phi'])
    
    arch_str = "arch: "
    arch_str += "(RNN) " if options['recurrent'] else ""
    arch_str += "[ "
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
   
    filename = ""
    filename += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename += "-emb{}".format(options['embedding_dim'])
    filename += "-acc{}".format(int(score*100))
    filename += "-gr" if options['evaluation_method'] == 'graph' else "-nn"
    filename += "-rnn" if options['recurrent'] else ""
    output_file = os.path.join(options['output_dir'], filename)
    
    export_results(output_file, model, fig, options)
    
    
if __name__ == "__main__":
    main()
