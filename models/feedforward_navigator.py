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

from common.preprocessing import *
from common.evaluation import *

##########################
# The feed-forward model #
##########################

def build_feedforward_model(embedding_dim,hidden_layers,activations,learning_rate):
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
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    return model


#################
# Main function #
#################

def main():    
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)
    
    ##########################
    # Standard configuration #
    ##########################
    
    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    embedding_dir = root_dir + "models/embeddings/"
    propagation_dir = root_dir + "logger/navigation_training/"
    
    params = {
        'embedding_file': embedding_dir + "20201201-142508-emb10-acc36",
        'detector_file': root_dir + "detector/detector_surfaces.csv",
        'propagation_file': propagation_dir + "data-201113-151732-n1.csv",
        'output_dir': root_dir + "models/navigator/",
        'export': True,
        'show': True,
        'test_split': 0.2,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'hidden_layers': [500,500,500],
        'activations': ['relu', 'relu', 'relu'],
        'batch_size': 256,
        'epochs': 200,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-l", "--learning_rate", type=float)
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-d", "--embedding_dim", type=int)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--layer_size", type=int, default=500)
    parser.add_argument("--network_depth", type=int, default=3)
    parser.add_argument("--test_split", type=float)
    parser.add_argument("--validation_split", type=float)
    parser.add_argument("--prop_data_events", type=int)
    parser.add_argument("--no_plt_show", action="store_true")
    parser.add_argument("--no_export",action="store_true")
    args = parser.parse_args()
    
    if args.no_export:
        params['export'] = False
        
    if args.no_plt_show:
        params['show'] = False
        
    if (args.layer_size and not args.network_depth) or (args.network_depth and not args.layer_size):
        raise Exception("'--layer_size' and '--network_depth' must be specified both or none of them!")

    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate

    if args.epochs is not None:
        params['epochs'] = args.epochs

    if args.test_split is not None:
        params['test_split'] = args.test_split

    if args.validation_split is not None:
        params['validation_split'] = args.validation_split
        
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
        
    if args.embedding_dim is not None:
        assert args.embedding_dim > 0
        
        suitable_embeddings = extract_embedding_models(embedding_dir)
        suitable_embeddings = [ e for e in suitable_embeddings if e[1] == args.embedding_dim ]
        suitable_embeddings.sort(key=lambda x: x[2], reverse=True)
        
        if len(suitable_embeddings) == 0:
            raise Exception("No suitable embedding found in '{}'".format(embedding_dir))
        
        params['embedding_file'] = suitable_embeddings[0][0]
       
       
    if args.prop_data_events is not None:
        assert args.prop_data_events > 0
        
        prop_datas = extract_propagation_data(propagation_dir)        
        prop_datas = [ d for d in prop_datas if d[1] == args.prop_data_events ]
        
        if len(prop_datas) == 0:
            raise Exception("No suitable propagation dataset found in '{}'".format(propagation_dir))
        
        params['propagation_file'] = prop_datas[0][0]
        
    
    assert args.layer_size > 0
    assert args.network_depth > 0
    
    params['hidden_layers'] = [ args.layer_size ] * args.network_depth
    params['activations'] = [ args.activation ] * args.network_depth
    
    assert params['validation_split'] > 0.0 and params['validation_split'] < 1.0
    assert params['test_split'] > 0.0 and params['test_split'] < 1.0
    assert params['epochs'] > 0
    assert params['learning_rate'] > 0.0
    assert params['batch_size'] > 0
    assert len(params['hidden_layers']) == len(params['activations'])
    
    logging.info("run with the following params:")
    pprint.pprint(params, width=1)

    ########################
    # Import training data #
    ########################
    
    x_embs, x_pars, y_embs, nn = prepare_data(params['propagation_file'], params['embedding_file'], params['detector_file'])
    
    embedding_dim = x_embs[0].shape[1]
    num_params = x_pars[0].shape[1]    
    
    # Train test split (includes shuffle)
    x_train_embs, x_test_embs, x_train_pars, x_test_pars, y_train_embs, y_test_embs = \
        train_test_split(x_embs, x_pars, y_embs, test_size=params['test_split'])
    
    # don't need track structure here
    x_train_embs = np.concatenate(x_train_embs)
    x_train_pars = np.concatenate(x_train_pars)
    y_train_embs = np.concatenate(y_train_embs)
    
    ###############################
    # Model building and Training #
    ###############################
    
    # Build model
    model_params = {
        'embedding_dim': embedding_dim,
        'hidden_layers': params['hidden_layers'],
        'activations': params['activations'],
        'learning_rate': params['learning_rate']
    }
    
    model = build_feedforward_model(**model_params)
    
    # Do the training    
    logging.info("start training")
    
    history = model.fit(
        x=[x_train_embs, x_train_pars], 
        y=y_train_embs,
        validation_split=params['validation_split'],
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=2
    )
    
    logging.info("finished training")  
    
    
    ##############
    # Evaluation #
    ##############
    
    track_lengths = np.array([ len(s) for s in x_test_embs ])
    
    y_true = np.concatenate(y_test_embs)
    y_pred = model([np.concatenate(x_test_embs),np.concatenate(x_test_pars)])
    
    fig, ax = make_evaluation_plot(y_true, y_pred.numpy(), nn, track_lengths, history)
    
    
    arch_str = "arch: [ "
    for size, activation in zip(params['hidden_layers'], params['activations']):
        arch_str += "({}, {}), ".format(size, activation)
    arch_str += "] "
    
    learning_rate_str = "lr: {} ".format(params['learning_rate'])
    epochs_str = "epochs: {} ".format(params['epochs'])
    
    fig.suptitle("Feedforward Network - " + arch_str + "- " + learning_rate_str + "- " + epochs_str, fontweight='bold')
    
    if params['show']:
        plt.show()

    res = neighbor_accuracy_detailed(y_true, y_pred.numpy(), nn)

    logging.info("neigbhor distribution (1-2-3-5-10-other): %.2f - %.2f - %.2f - %.2f - %.2f - %.2f", *res)
       
    ##########   
    # Export #
    ##########
   
    if params['export']: 
        method_string = "-forward"
        date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        embedding_string = "-emb" + str(embedding_dim)
        accuracy_string = "-acc" + str(int(round(next_neighbor_accuracy(y_true, y_pred, nn)*100)))
        output_filename = date_string + method_string + embedding_string + accuracy_string
    
        model.save(params['output_dir'] + output_filename)
        logging.info("exported model to '" + params['output_dir'] + output_filename + "'")
    
        fig.savefig(params['output_dir'] + output_filename + ".png")
        logging.info("exported chart to '" + params['output_dir'] + output_filename + ".png" + "'")
    
    
if __name__ == "__main__":
    main()
