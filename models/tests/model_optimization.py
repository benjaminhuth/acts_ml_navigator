import os
import datetime
import logging

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import kerastuner as kt

from feedforward_navigator import build_feedforward_model
from common.preprocessing import prepare_data
from sklearn.model_selection import train_test_split


def build_model(hp, embedding_dim):
    
    layers = [ hp.Int('layer_size', 100, 2000) ] * hp.Int('network_depth', 1, 10)
    activations = [ hp.Choice("act_" + str(i), ['relu', 'tanh']) for i in range(len(layers)) ]
        
    model_params = {
        'embedding_dim': embedding_dim,
        'hidden_layers': layers,
        'activations': activations,
        'learning_rate': hp.Choice('learning_rate', [1.0, 0.1, 0.01, 0.001, 0.0001]),
    }
    
    return build_feedforward_model(**model_params)
    
    
def main():
    #########################
    # General configuration #
    #########################
    
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        
    logging.info("tensorflow devices: %s",[ t for n, t in tf.config.list_physical_devices() ])
    
    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"

    params = {
        'embedding_file': root_dir + "models/embeddings/20201201-142508-emb10-acc36",
        'detector_file': root_dir + "detector/detector_surfaces.csv",
        'propagation_file': root_dir + "logger/navigation_training/data-201113-151732-n1.csv",
        'test_split': 0.3,
    }
    
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
    
    x_test_embs = np.concatenate(x_test_embs)
    x_test_pars = np.concatenate(x_test_pars)
    y_test_embs = np.concatenate(y_test_embs)
    
    ##################
    # Optimize model #
    ##################
    
    method_string = "-forward"
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_name = "feedforward_model"
    project_dir = root_dir + "optimization/"
    
    tuner = kt.tuners.Hyperband(
        lambda hp: build_model(hp, embedding_dim),
        objective = 'val_loss',
        max_epochs = 200,
        factor = 3,
        directory = project_dir,
        project_name = project_name
    )
    
    tuner.search(
        x = [x_train_embs, x_train_pars],
        y = y_train_embs, 
        epochs = 200, 
        validation_data=([x_test_embs, x_test_pars], y_test_embs),
        batch_size = 256,
        verbose = 2,
        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    print(best_hps)
    
    
 
 
if __name__ == "__main__":
    main()
