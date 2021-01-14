import os
import datetime
import logging

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
from common.plot_embedding import *
from common.pairwise_score_tools import *



###################
# Build the model #
###################

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
        run_eagerly=True
    )
    
    return model



def evaluate_edge(pos, start_emb, target_emb, x_params, score_matrix, graph_edge_map, nn, nav_model, emb_model):
    '''
    
    '''    
    start_id = int(np.squeeze(nn.kneighbors(start_emb.reshape(1,-1), 1, return_distance=False)))
    target_ids = graph_edge_map[ start_id ].targets
    target_embs = np.squeeze(emb_model(target_ids))
    assert target_emb in target_embs
        
    # Broadcast input values to arrays for vectorized inference
    starts = np.full(len(target_ids), start_emb )
    params = np.full((len(target_ids),len(x_params)), x_params)
    
    scores = np.squeeze(nav_model([starts, target_embs, params]))
    score_idxs = np.flip(np.argsort(scores))
    
    # Sort targets by predicted score
    target_embs = target_embs[score_idxs]
    
    # Find where in the list the correct result is
    correct_pos = int(np.argwhere(np.equal(target_embs,target_emb)))
    
    # Fill abs_score_matrix
    if correct_pos == 0: score_matrix.loc[pos, 'in1'] += 1
    elif correct_pos == 1: score_matrix.loc[pos, 'in2'] += 1
    elif correct_pos == 2: score_matrix.loc[pos, 'in3'] += 1
    elif correct_pos < 5: score_matrix.loc[pos, 'in5'] += 1
    elif correct_pos < 10: score_matrix.loc[pos, 'in10'] += 1
    else: score_matrix.loc[pos, 'other'] += 1
    
    # Fill res_scores (do the 1- to let the best result be 1)
    score_matrix.loc[pos, 'relative_score'] += 1 - correct_pos/len(target_embs)
    score_matrix.loc[pos, 'num_edges'] += len(target_embs)
    
    return score_matrix

#################
# Main function #
#################

def main():
    options = init_options_and_logger(get_navigation_training_dir(),
                                      get_root_dir() + "models/pairwise_score_navigator/")
    
    embedding_dir = os.path.join(get_root_dir(), 'models/target_pred_navigator/embeddings/')
    embedding_info = extract_embedding_model(embedding_dir, options['embedding_dim'])
    options['embedding_file'] = embedding_info.path
    options['beampipe_split_z'] = embedding_info.bpsplit_z
    options['beampipe_split_phi'] = embedding_info.bpsplit_phi
    
    logging.info("imported embedding '%s' with beampipe split (%d,%d)", 
                 options['embedding_file'], options['beampipe_split_z'], options['beampipe_split_phi'])
    
    ################
    # PREPARE DATA #
    ################
        
    prop_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64}) 
    
    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    # Beampipe split and new mapping
    prop_data = beampipe_split(prop_data, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data = geoid_to_ordinal_number(prop_data, detector_data, total_beampipe_split)
    
    # Categorize into tracks (also needed for testing later)
    selected_params = ['dir_x', 'dir_y', 'dir_z', 'qop']
    train_starts, train_params, train_targets = \
        categorize_into_tracks(prop_data, total_beampipe_split, selected_params)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(train_starts), max([ len(track) for track in train_starts]))
    
    
    # Load embedding model
    embedding_model = tf.keras.models.load_model(options['embedding_file'], compile=False)
    assert options['embedding_dim'] == np.squeeze(embedding_model(0)).shape[0]
    
    for i in range(len(train_starts)):
        train_starts[i] = np.squeeze(embedding_model(train_starts[i]))
        train_targets[i] = np.squeeze(embedding_model(train_targets[i]))
    
    # Split train test
    train_starts, test_starts, train_params, test_params, train_targets, test_targets = \
        train_test_split(train_starts, train_params, train_targets, test_size=options['test_split'])
    
    # Merge start and end id for the fit function
    train_edges = []
    for start_ids, end_ids in zip(train_starts, train_targets):
        train_edges.append( np.stack([start_ids, end_ids], axis=1) )
    
    test_edges = []
    for start_ids, end_ids in zip(test_starts, test_targets):
        test_edges.append( np.stack([start_ids, end_ids], axis=1) )
    
    ###############
    # Train model #
    ###############   
    
    # Prepare training and test/validation generator
    train_gen = gen_batch_shuffle(options['batch_size'], np.concatenate(train_edges),
                           np.concatenate(train_params), total_node_num)
    
    test_gen = gen_batch_shuffle(2*len(np.concatenate(test_edges)), np.concatenate(test_edges), 
                          np.concatenate(test_params), total_node_num)
        

    # Build model
    model_params = {
        'embedding_dim': options['embedding_dim'],
        'layers': [ options['layer_size'] ] * options['network_depth'],
        'activations': [ 'relu' ] * options['network_depth'],
        'learning_rate': options['learning_rate']
    }
    
    navigation_model = build_model(**model_params)
    
    # Build graph (needed for testing/scoring)
    graph_edge_map = generate_graph_edge_map(prop_data, total_node_num)
    
    # Do training
    steps_per_epoch = len(np.concatenate(train_edges)) // options['batch_size']
    logging.info("Start learning embedding, steps_per_epoch = %d", steps_per_epoch)
    
    history = navigation_model.fit(
        x=train_gen, 
        steps_per_epoch=steps_per_epoch,
        epochs=options['epochs'],
        validation_data=next(test_gen),
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            RemainingTimeEstimator(options['epochs']),
        ]
    )
    
    ##############
    # Evaluation #
    ##############
    
    logging.info("start evaluation with %d tracks", len(test_edges));
    
    # this is WIP, similarly to the pretrained target_pred_navigator there must be ensured that the nn matches the embedding model 
    # TODO apply improvements from target_pred_navigator_pre (maybe again copy large parts 
    # TODO Don't forget improvements of export and JSON export for configurations...
        
    fig, axes, score = \
        make_evaluation_plots(test_starts, test_targets, test_params, history.history,
                              lambda a,b,c,d,e: evaluate_edge(a,b,c,d,e,graph_edge_map,navigation_model,nn,embedding_model))
        
    # Summary title and data info 
    data_gen_str = "gen: random" if options['sample_gen_method'] == 'random' else "gen: shuffle"
    bpsplit_str = "bp split: z={}, phi={}".format(options['beampipe_split_z'],options['beampipe_split_phi'])
    
    arch_str = "arch: [ "
    for size, activation in zip(model_params['layers'], model_params['activations']):
        arch_str += "({}, {}), ".format(size, activation)
    arch_str += "]"
    
    lr_str = "lr: {}".format(options['learning_rate'])
    
    fig.suptitle("Pairwise Score Nav: {} - {} - {} - {}".format(bpsplit_str, data_gen_str, arch_str, lr_str), fontweight='bold')
    fig.text(0,0,"Training data: " + options['propagation_file'])
    
    plt.show()
    
    ##########
    # Export #
    ##########
    
    date_str   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    method_str = "-rnd" if options['sample_gen_method'] == 'random' else '-shf'
    size_str   = "-n{}".format(options['prop_data_size'])
    emb_str    = "-emb{}".format(options['embedding_dim'])
    acc_str    = "-acc{}".format(round(score*100))
    output_filename = date_str + emb_str + method_str + size_str + acc_str

    if options['export']:
        navigation_model.save(options['output_dir'] + output_filename)
        logging.info("exported model to '" + options['output_dir'] + output_filename + "'")

        fig.savefig(options['output_dir'] + output_filename + ".png")
        logging.info("exported chart to '" + options['output_dir'] + output_filename + ".png" + "'")
    else:
        logging.info("output filename would be: '%s'",output_filename)
    
    
    # Create separate embedding model for plotting with reduced dimensionality
    #input_node = tf.keras.Input(1)
    #output_node = model.get_layer("embedding")(input_node)
    
    #embedding_model = tf.keras.Model(input_node,output_node)
    
    #plot_embedding(embedding_model, params['detector_file'], 3)
    
    
        
if __name__ == "__main__":
    main()

