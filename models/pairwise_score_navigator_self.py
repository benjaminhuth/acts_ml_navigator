import os
import datetime
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '12'  # does this speed things up?
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from common.preprocessing import *
from common.evaluation import *
from common.misc import *
from common.plot_embedding import *
from common.pairwise_score_tools import *

###############
# Build model #
###############
    
def build_model(num_categories, embedding_dim, layers, activations, learning_rate):
    '''
    Function that builds and compiles the model. It is of form:
    
    [start_id, target_id, start_params] -> [0,1]
    
    Parameters:
    * num_categories (int): how many ids for the embedding layer
    * embedding_dim (int)
    * layers (list): list of layer sizes
    * activations (list): list of activations
    * learning_rate (float)
    
    Returns:
    * compiled Keras model
    '''
    input_node_1 = tf.keras.Input(1)
    input_node_2 = tf.keras.Input(1)
    input_params = tf.keras.Input(4)
    
    embedding_layer = tf.keras.layers.Embedding(num_categories,embedding_dim)
    
    a = embedding_layer(input_node_1)
    a = tf.keras.layers.Reshape((embedding_dim,))(a)
    
    b = embedding_layer(input_node_2)
    b = tf.keras.layers.Reshape((embedding_dim,))(b)
    
    c = tf.keras.layers.Dense(embedding_dim)(input_params)
    
    d = tf.keras.layers.Concatenate()([ a, b, c ])
    
    d = tf.keras.layers.Dense(layers[0],activation=activations[0])(d)
    
    for i in range(len(layers)-1):
        d = tf.keras.layers.Dense(layers[0],activation=activations[0])(d)
        
    output = tf.keras.layers.Dense(1, activation="sigmoid")(d)
    
    model = tf.keras.Model(inputs=[input_node_1,input_node_2,input_params],outputs=[output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy()],
        #run_eagerly=True
    )
    
    return model
   

    
def evaluate_edge(pos, start, target, param, result, nav_model, graph_map):
    '''
    Function which evaluates all metrics for a single edge in a track.
    
    Parameters:
    * pos: the position in the track (int)
    * start: a id representing the start surface
    * target: a id representing the target surface
    * param: the track parameters at the start (i.e. direction and qop)
    * result: named tuple EvaluationResult
    * nav_model: the navigation model to predict the score of a tuple [start_surface, end_surface, params]
    * graph_edge_map: a dictionary { start: namedtuple( targets, weights ) }
    
    Returns:
    * modified score_matrix (pandas dataframe)
    '''
    node_data = graph_map[ start ]
    targets = node_data.targets
        
    # Broadcast input values to arrays for vectorized inference
    starts = np.full(len(targets), start )
    params = np.full((len(targets),len(param)), param)
    
    scores = np.squeeze(nav_model([starts, targets, params]))
    score_idxs = np.flip(np.argsort(scores))
    
    # Sort targets by predicted score
    targets = targets[score_idxs]
    
    # Find where in the list the correct result is
    correct_pos = int(np.argwhere(np.equal(targets,target)))
    
    # Fill abs_score_matrix
    return fill_in_results(pos, correct_pos, node_data.position[2], result, len(targets))


#################
# Main function #
#################

def main():
    options = init_options_and_logger(get_navigation_training_dir(),
                                      os.path.join(get_root_dir(), "models/pairwise_score_navigator_self/"))
    
    options['propagation_file_false'] = extract_propagation_data(get_false_samples_dir())[ options['prop_data_size'] ]
    logging.info("Load false samples from '%s'", options['propagation_file_false'])
    logging.warning(">>> Does not yet support false samples method graph 'graph' <<<)")
    
    ################
    # PREPARE DATA #
    ################
        
    prop_data_true = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    prop_data_false = pd.read_csv(options['propagation_file_false'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64}) 
    
    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    selected_params = ['dir_x', 'dir_y', 'dir_z', 'qop']
    
    prop_data_true = uniform_beampipe_split(prop_data_true, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data_true = geoid_to_ordinal_number(prop_data_true, detector_data, total_beampipe_split)
    true_tracks = categorize_into_tracks(prop_data_true, total_beampipe_split, selected_params)
    
    prop_data_false = uniform_beampipe_split(prop_data_false, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data_false = geoid_to_ordinal_number(prop_data_false, detector_data, total_beampipe_split)
    false_tracks = categorize_into_tracks(prop_data_false, total_beampipe_split, selected_params)
    
    logging.info("Imported %d TRUE tracks, the maximum sequence length is %d",
                 len(true_tracks[0]), max([ len(track) for track in true_tracks[0]]))
    logging.info("Imported %d FALSE tracks, the maximum sequence length is %d",
                 len(false_tracks[0]), max([ len(track) for track in false_tracks[0]]))
    
    train_starts, test_starts, train_params, test_params, train_targets, test_targets = \
        train_test_split(true_tracks.start_ids, true_tracks.start_params, true_tracks.target_ids,
                         test_size=options['test_split'])
    
    num_true_train_samples = len(np.concatenate(train_starts))
    num_false_train_samples = len(np.concatenate(false_tracks.start_ids))
    
    # Add false samples
    train_starts += false_tracks.start_ids
    train_params += false_tracks.start_params
    train_targets += false_tracks.target_ids
    
    # Flatten out track structure for training data
    train_starts = np.concatenate(train_starts)
    train_params = np.concatenate(train_params)
    train_targets = np.concatenate(train_targets)
    train_y = np.concatenate([ np.ones(num_true_train_samples), np.zeros(num_false_train_samples) ])
    
    assert len(train_starts) == len(train_params) == len(train_targets) == len(train_y)
    
    logging.info("Training data: %.2f%% true samples, %.2f%% false samples",
                 100 * num_true_train_samples / (num_true_train_samples + num_false_train_samples),
                 100 * num_false_train_samples / (num_true_train_samples + num_false_train_samples))
    
    ###############
    # Train model #
    ###############   
    
    # Build model
    model_params = {
        'num_categories': total_node_num,
        'embedding_dim': options['embedding_dim'],
        'layers': [ options['layer_size'] ] * options['network_depth'],
        'activations': [ 'relu' ] * options['network_depth'],
        'learning_rate': options['learning_rate']
    }
    
    model = build_model(**model_params)
    #model.summary()
    
    logging.info("Start learning embedding")
    
    history = model.fit(
        x=[train_starts, train_targets, train_params],
        y=train_y, 
        batch_size=options['batch_size'],
        epochs=options['epochs'],
        validation_split=options['validation_split'],
        verbose=2,
        callbacks=[
            #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            RemainingTimeEstimator(options['epochs']),
        ]
    )
    
    ##############
    # Evaluation #
    ##############
    
    logging.info("start evaluation with %d tracks", len(test_starts));
    
    # Build graph (needed for testing/scoring)
    graph_map = make_graph_map(prop_data_true, total_node_num)
        
    fig, axes, score = \
        evaluate_and_plot(test_starts, test_params, test_targets, history.history,
                              lambda a,b,c,d,e: evaluate_edge(a,b,c,d,e,model,graph_map))
        
    # Summary title and data info 
    data_gen_str = "gen: simulated"
    bpsplit_str = "bp split: z={}, phi={}".format(options['beampipe_split_z'],options['beampipe_split_phi'])
    
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
    method_str = "-gen"
    size_str   = "-n{}".format(options['prop_data_size'])
    emb_str    = "-emb{}".format(options['embedding_dim'])
    acc_str    = "-acc{}".format(round(score*100))
    output_file = os.path.join( options['output_dir'], date_str + emb_str + method_str + size_str + acc_str )
    
    export_results(output_file, model, fig, options)
    
    
    # Create separate embedding model for plotting with reduced dimensionality
    #input_node = tf.keras.Input(1)
    #output_node = model.get_layer("embedding")(input_node)
    
    #embedding_model = tf.keras.Model(input_node,output_node)
    
    #plot_embedding(embedding_model, params['detector_file'], 3)
    
    
        
if __name__ == "__main__":
    main()
