import os
import datetime
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from common.preprocessing import *
from common.evaluation import *
from common.misc import *
from common.pairwise_score_tools import *



def build_model(embedding_dim, layers, activations, learning_rate):
    '''
    Function that builds and compiles the model. It is of form:
    
    [start_emb, target_emb, start_params] -> [0,1]
    
    Parameters:
    * embedding_dim (int)
    * layers (list): list of layer sizes
    * activations (list): list of activations
    * learning_rate (float)
    
    Returns:
    * compiled Keras model
    '''
    start_emb = tf.keras.Input(embedding_dim)
    target_emb = tf.keras.Input(embedding_dim)
    track_params = tf.keras.Input(4)
    
    d = tf.keras.layers.Concatenate()([ start_emb, target_emb, track_params ])
    
    d = tf.keras.layers.Dense(layers[0],activation=activations[0])(d)
    
    for i in range(len(layers)-1):
        d = tf.keras.layers.Dense(layers[0],activation=activations[0])(d)
        
    output = tf.keras.layers.Dense(1, activation="sigmoid")(d)
    
    model = tf.keras.Model(inputs=[start_emb,target_emb,track_params],outputs=[output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy()],
        #run_eagerly=True
    )
    
    return model



def evaluate_edge(pos, start_emb, target_emb, x_params, result,
                  graph_map, nn, nav_model, emb_model):
    '''
    Evaluates one edge and puts result into a score_matrix. Evaluation is based on the graph, but since the model works with pretrained embeddings, we need the nn-index to retrieve the embeddings of the graph items. Therefore the embedding_model and the nn-index MUST match exactly.
    
    Parameters:
    * pos (int): position in track
    * x_emb (ndarray): start embedding
    * y_true (ndarray): true target embedding
    * x_params (ndarray): track parameters at start
    * result: named tuple EvaluationResult
    * graph_map: a dictionary { id: (targets, weights, real space position) }
    * surface_pos_map: a dictionary { id: real_space_position }
    * nn (sklearn knn index)
    * nav_model: Keras model ([start_id, end_id, params] -> score [0,1])
    * emb_model: Keras model (id -> embedding)
    
    Returns:
    * dataframe: the modified score_matrix
    '''    
    assert start_emb.shape == target_emb.shape
    assert len(start_emb.shape) == len(x_params.shape) == 1
    
    # Get ID and data of the current graph node
    start_id = int(np.squeeze(nn.kneighbors(start_emb.reshape(1,-1), 1, return_distance=False)))
    node_data = graph_map[ start_id ]
    
    # Get target embeddings with embedding_model
    target_ids = node_data.targets
    target_embs = np.squeeze(emb_model(target_ids),axis=1)
    assert target_emb in target_embs
        
    # Broadcast input values to arrays for vectorized inference
    broadcasted_start = np.full((len(target_ids), len(start_emb)), start_emb)
    params = np.full((len(target_ids),len(x_params)), x_params)
    
    scores = np.squeeze(nav_model([broadcasted_start, target_embs, params]))
    score_idxs = np.flip(np.argsort(scores))
    
    # Sort targets by predicted score
    target_embs = target_embs[score_idxs]
    
    # Find where in the list the correct result is    
    correct_pos = int(np.argwhere((target_embs == target_emb).all(axis=1)))
    
    return fill_in_results(pos, correct_pos, node_data.position[2], result, len(target_ids))



def generate_train_data_with_false_simulated(true_samples_tracks, false_samples_file, bpsplit_z, 
                                             bpsplit_phi, detector_data, selected_params):
    '''
    Generates training data from true data and a set of simulated false data. Removes the track structure in the data
    
    Parameters:
    * true_samples_tracks: named tuple of type TrackCollection
    * false_samples_file: filename where the false samples are stored
    * bpsplit_z: int
    * bpsplit_phi: int
    * detector_data: pd dataframe
    * selected_params: list of strings, e.g. ['dir_x','dir_y','dir_z']
    
    Returns:
    * a tuple: ([start_ids, target_ids, start_params], y)
    '''
    assert type(true_samples_tracks).__name__ == 'TrackCollection'
    assert bpsplit_phi > 0
    assert type(bpsplit_z) == list and len(bpsplit_z) > 1
    
    total_beampipe_split = (len(bpsplit_z)-1)*bpsplit_phi
    
    # Import false samples
    prop_data_false = pd.read_csv(false_samples_file, dtype={'start_id': np.uint64, 'end_id': np.uint64})
    prop_data_false, dist = apply_beampipe_split(prop_data_false, bpsplit_z, bpsplit_phi, return_z_distribution=True)
    
    logging.info("False sim data - beampipe split info: mean=(%.2f +- %.2f), max=%.2f, min=%.2f", 
                 np.mean(dist[1]), np.std(dist[1]), np.amax(dist[1]), np.amin(dist[1]))
    
    prop_data_false = geoid_to_ordinal_number(prop_data_false, detector_data, total_beampipe_split)
    false_samples_tracks = categorize_into_tracks(prop_data_false, total_beampipe_split, selected_params)
    
    logging.info("Imported %d FALSE tracks, the maximum sequence length is %d",
                 len(false_samples_tracks[0]), max([ len(track) for track in false_samples_tracks[0]]))
        
    # Number of samples
    num_true_train_samples = len(np.concatenate(true_samples_tracks.start_ids))
    num_false_train_samples = len(np.concatenate(false_samples_tracks.start_ids))
    
    # Combine samples
    train_starts = true_samples_tracks.start_ids + false_samples_tracks.start_ids
    train_params = true_samples_tracks.start_params + false_samples_tracks.start_params
    train_targets = true_samples_tracks.target_ids + false_samples_tracks.target_ids
    
    # Flatten out track structure for training data
    train_starts = np.concatenate(train_starts)
    train_params = np.concatenate(train_params)
    train_targets = np.concatenate(train_targets)
    train_y = np.concatenate([ np.ones(num_true_train_samples), np.zeros(num_false_train_samples) ])
    
    assert len(train_starts) == len(train_params) == len(train_targets) == len(train_y)
    return [ train_starts, train_targets, train_params ], train_y



def generate_train_data_from_graph(true_samples_tracks, graph_edge_map, max_false_per_true):
    '''
    This funciton generates training data from the graph, which is exactly what's later tested in the evaluation.
    TODO Think about sophisticated rebalancing, for now only take one false sample per true sample
    
    Parameters:
    * true_samples_tracks: named tuple of type TrackCollection
    * graph_edge_map: dictionary
    
    Returns:
    * a tuple: ([start_ids, target_ids, start_params], y)
    '''
    
    true_starts = np.concatenate(true_samples_tracks.start_ids)
    true_params = np.concatenate(true_samples_tracks.start_params)
    true_targets = np.concatenate(true_samples_tracks.target_ids)
    
    false_starts = []
    false_params = []
    false_targets = []
    
    for start_id, true_target, params in zip(true_starts, true_targets, true_params):
        targets = graph_edge_map[ start_id ].targets
        
        true_idx = int(np.argwhere(np.equal(targets, true_target)))
        
        if len(targets) < 2:
            continue
        
        elif len(targets) == 2:
            false_starts.append(start_id)
            false_params.append(params)
            false_targets.append(targets[true_idx-1])
        
        else:
            num_false = min( len(targets)-1, max_false_per_true )
            
            other_idxs = true_idx - (np.arange(0, len(targets)-1) + 1)
            false_idxs = np.random.choice(other_idxs, num_false)
            
            for idx in false_idxs:
                assert true_target != targets[idx]
                
                false_starts.append(start_id)
                false_params.append(params)
                false_targets.append(targets[idx])

        
    false_starts = np.array(false_starts)
    false_params = np.array(false_params)
    false_targets = np.array(false_targets)
    
    
    # Form final training data
    train_starts = np.concatenate([ true_starts, false_starts ])
    train_params = np.concatenate([ true_params, false_params ])
    train_targets = np.concatenate([ true_targets, false_targets ])
    train_y = np.concatenate([ np.ones(len(true_targets)), np.zeros(len(false_targets)) ])
    
    assert len(train_starts) == len(train_params) == len(train_targets) == len(train_y)
    
    logging.info("Generated false samples, true/false ratio: %.2f", len(true_starts)/len(false_starts))
    return [train_starts, train_targets, train_params], train_y
        
    
        





#####################
# The main function #
#####################

def main():
    options = init_options_and_logger(get_navigation_training_dir(),
                                      os.path.join(get_root_dir(), "models/pairwise_score_navigator_pre/"),
                                      { 'data_gen_method': 'graph', 'graph_gen_false_per_true': 2 })
    
    assert options['data_gen_method'] == 'graph' or options['data_gen_method'] == 'false_sim'
    
    # False propagation data
    if options['data_gen_method'] == 'false_sim':
        options['propagation_file_false'] = extract_propagation_data(get_false_samples_dir())[ options['prop_data_size'] ]
        logging.info("Load false samples from '%s'", options['propagation_file_false'])
    
    # Embedding import
    embedding_dir = os.path.join(get_root_dir(), 'models/embeddings/')
    embedding_info = extract_embedding_model(embedding_dir, options['embedding_dim'], options['bpsplit_z'],
                                             options['bpsplit_phi'], options['bpsplit_method'])
    
    options['embedding_file'] = embedding_info.path
    options['bpsplit_z'] = embedding_info.bpsplit_z
    options['bpsplit_phi'] = embedding_info.bpsplit_phi
    
    logging.info("imported embedding '%s' with beampipe split (%d,%d)", 
                 os.path.basename(options['embedding_file']), len(options['bpsplit_z'])-1, options['bpsplit_phi'])
    
    
    ################
    # PREPARE DATA #
    ################
        
    # Detector and beampipe split
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64}) 
    
    total_beampipe_split = (len(options['bpsplit_z'])-1)*options['bpsplit_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    selected_params = ['dir_x', 'dir_y', 'dir_z', 'qop']
    
    # True samples
    prop_data_true = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    
    # Beampipe split
    prop_data_true, z_dist = apply_beampipe_split(prop_data_true, options['bpsplit_z'], 
                                                  options['bpsplit_phi'], return_z_distribution=True)
    
    logging.info("beampipe split info: mean=(%.2f +- %.2f), max=%.2f, min=%.2f", np.mean(z_dist[1]),
                 np.std(z_dist[1]), np.amax(z_dist[1]), np.amin(z_dist[1]))
    
    # Plot and save z bin distribution
    plt.plot(z_dist[0], z_dist[1], ".-")
    plt.xlabel("z-bins")
    plt.ylabel("# of surfaces on z-bin")
    plt.title("Surfaces per z-bin (z-split = {}, n={})".format(len(options['bpsplit_z'])-1, options['prop_data_size']))
    
    if options['show']:
        plt.show()
        
    # Do the rest of preprocessing
    prop_data_true = geoid_to_ordinal_number(prop_data_true, detector_data, total_beampipe_split)
    true_tracks = categorize_into_tracks(prop_data_true, total_beampipe_split, selected_params)
        
    logging.info("Imported %d TRUE tracks, the maximum sequence length is %d",
                 len(true_tracks.start_ids), max([ len(track) for track in true_tracks.start_ids]))
    
    train_tracks, test_tracks = split_in_train_and_test_set(true_tracks, options['test_split'])
    
    # Build graph (needed for testing/scoring/ data generation)
    graph_map = make_graph_map(prop_data_true, total_node_num)
    
    # Generate training data structure
    x_train = y_train = None
    
    if options['data_gen_method'] == 'false_sim':
        x_train, y_train = generate_train_data_with_false_simulated(train_tracks, options['propagation_file_false'],
                                                                    options['bpsplit_z'], options['bpsplit_phi'],
                                                                    detector_data, selected_params)
    else:
        x_train, y_train = generate_train_data_from_graph(train_tracks, graph_map, options['graph_gen_false_per_true'])    
    
    # Shuffle
    idxs = np.arange(len(y_train))
    np.random.shuffle(idxs)
    
    x_train[0] = x_train[0][idxs]   # start ids
    x_train[1] = x_train[1][idxs]   # target ids
    x_train[2] = x_train[2][idxs]   # start params 
    y_train = y_train[idxs]         # 0 or 1
    
    assert x_train[1].all() >= 0
    assert x_train[0].all() >= 0
    
    # Apply embedding
    embedding_model = tf.keras.models.load_model(options['embedding_file'], compile=False)
    assert options['embedding_dim'] == np.squeeze(embedding_model(0)).shape[0]
    
    x_train[0] = np.squeeze(embedding_model(x_train[0]))
    x_train[1] = np.squeeze(embedding_model(x_train[1]))
    
    logging.info("Total number of samples: %d", len(y_train))
    
    ###############
    # Train model #
    ###############   

    # Build model
    model_params = {
        'embedding_dim': options['embedding_dim'],
        'layers': [ options['layer_size'] ] * options['network_depth'],
        'activations': [ 'relu' ] * options['network_depth'],
        'learning_rate': options['learning_rate']
    }
    
    navigation_model = build_model(**model_params)
    
    logging.info("Start training")
    
    history = navigation_model.fit(
        x=x_train,
        y=y_train,
        batch_size=options['batch_size'],
        epochs=options['epochs'],
        validation_split=options['validation_split'],
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            RemainingTimeEstimator(options['epochs']),
        ]
    )
    
    ##############
    # Evaluation #
    ##############
    
    logging.info("Start evaluation with %d tracks", len(test_tracks.start_ids));
    
    test_start_ids = [ np.squeeze(embedding_model(ids)) for ids in test_tracks.start_ids ]
    test_target_ids = [ np.squeeze(embedding_model(ids)) for ids in test_tracks.target_ids ]
    
    # Build nn index matching the embedding model
    nn = NearestNeighbors()
    nn.fit(np.squeeze(embedding_model(np.arange(total_node_num))))
    assert nn_index_matches_embedding_model(embedding_model, nn)
        
    fig, axes, score = \
        evaluate_and_plot(test_start_ids, test_tracks.start_params, test_target_ids, history.history,
                              lambda a,b,c,d,e: evaluate_edge(a,b,c,d,e,graph_map,nn,navigation_model,embedding_model))
        
    # Summary title and data info 
    data_gen_str = "gen: simulated" if options['data_gen_method'] == 'false_sim' else "gen: graph ({})".format(options['graph_gen_false_per_true'])
    
    bpsplit_str = "bp split: z={}, phi={}".format(len(options['bpsplit_z'])-1,options['bpsplit_phi'])
    
    arch_str = "arch: [ "
    for size, activation in zip(model_params['layers'], model_params['activations']):
        arch_str += "({}, {}), ".format(size, activation)
    arch_str += "]"
    
    lr_str = "lr: {}".format(options['learning_rate'])
    
    fig.suptitle("Pairwise Score Nav: {} - {} - {} - {}".format(bpsplit_str, data_gen_str, arch_str, lr_str), fontweight='bold')
    fig.text(0,0,"Training data: " + options['propagation_file'])
    
    if options['show']:
        plt.show()
    
    ##########
    # Export #
    ##########
    
    date_str   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    method_str = "-graph" if options['data_gen_method'] == 'graph' else "-sim"
    size_str   = "-n{}".format(options['prop_data_size'])
    emb_str    = "-emb{}".format(options['embedding_dim'])
    acc_str    = "-acc{}".format(round(score*100))
    output_file = os.path.join(options['output_dir'], date_str + emb_str + method_str + size_str + acc_str)

    export_results(output_file, navigation_model, fig, options)    
    
        
if __name__ == "__main__":
    main()

