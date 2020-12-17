import os
import sys
import argparse
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
from common.plotting import *


#######################
# The recurrent model #
#######################

def build_recurrent_model(embedding_dim, num_params, timesteps, dense_layers, recurrent_layers, learning_rate):
    assert len(dense_layers) >= 1 and len(recurrent_layers) >= 1
    
    input_embs = tf.keras.Input((timesteps, embedding_dim))
    input_pars = tf.keras.Input((timesteps, num_params))
    
    a = tf.keras.layers.Concatenate(axis=2)([input_pars,input_embs])
    
    for layer_size in recurrent_layers:
        a = tf.keras.layers.GRU(layer_size,return_sequences=True)(a)
    
    for layer_size in dense_layers:
        a = tf.keras.layers.Dense(layer_size)(a)
        a = tf.keras.layers.Activation(tf.nn.relu)(a)
    
    output_embs = tf.keras.layers.Dense(embedding_dim)(a)
    
    model = tf.keras.Model(inputs=[input_pars, input_embs], outputs=[output_embs])
    
    #model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    return model


#####################
# The main function #
#####################

def main(argv):
    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    propagation_file = root_dir + "logger/navigation_training/data-201113-151732-n1.csv"
    embedding_file = root_dir + "models/embeddings/20201201-142508-emb10-acc36"
    detector_file = root_dir + "detector/detector_surfaces.csv"
    
    #################
    # Setup logging #
    #################
    
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)
    
    ###############
    # Import data #
    ###############
    
    x_embs, x_pars, y_embs, nn = prepare_data(propagation_file, embedding_file, detector_file)
    
    embedding_dim = x_embs[0].shape[1]
    num_params = x_pars[0].shape[1]
    
    # Do train-test split before padding, so we can test performance of un-padded samples
    test_split = 0.2
    x_emb_train, x_emb_test, x_par_train, x_par_test, y_emb_train, y_emb_test = \
        train_test_split(x_embs, x_pars, y_embs, test_size=test_split)
    
    ###################################
    # Do pre-padding on training data #
    ###################################
    
    max_sequence_length = max([ len(track) for track in x_emb_train ])
    
    for i in range(len(x_emb_train)):
        if len(x_emb_train[i]) < max_sequence_length:
            pad = max_sequence_length - len(x_emb_train[i])
            
            x_emb_train[i] = np.concatenate([np.zeros((pad,embedding_dim)), x_emb_train[i]])
            x_par_train[i] = np.concatenate([np.zeros((pad,num_params)), x_par_train[i]])
            y_emb_train[i] = np.concatenate([np.zeros((pad,embedding_dim)), y_emb_train[i]])
            
    logging.info("after padding: x_emb_train[0].shape: %s", x_emb_train[0].shape)
    logging.info("after padding: x_par_train[0].shape: %s", x_par_train[0].shape)
    logging.info("after padding: y_emb_train[0].shape: %s", y_emb_train[0].shape)    
    
    #############################
    # Reformat and prepare data #
    #############################
            
    x_train = [ np.asarray(x_emb_train).astype(np.float32), np.asarray(x_par_train).astype(np.float32) ]
    y_train = np.asarray(y_emb_train).astype(np.float32)
    
    assert len(x_train[0]) == len(x_train[1]) == len(y_train)
    
    model_params = {
        'embedding_dim': embedding_dim,
        'num_params': num_params,
        'timesteps': max_sequence_length,
        'dense_layers': [500],
        'recurrent_layers': [500,500],
        'learning_rate': 0.001
    }
    
    model = build_recurrent_model(**model_params)
    
    history = model.fit(
        x=x_train, 
        y=y_train,
        validation_split=0.2,
        batch_size=256,
        epochs=1,
        verbose=2,
    )
    
    logging.info("Training finished")
    
    ############################
    # Test with un-padded data #
    ############################
    
    y_emb_pred = []
    
    for x_emb, x_par in zip(x_emb_test,x_par_test):
        x_emb = np.reshape(x_emb,newshape=(1,x_emb.shape[0],x_emb.shape[1]))
        x_par = np.reshape(x_par,newshape=(1,x_par.shape[0],x_par.shape[1]))
        y_emb_pred.append(np.squeeze(model([x_emb, x_par])))
        
    y_true = np.concatenate(y_emb_test)
    y_pred = np.concatenate(y_emb_pred)
    
    exit()
    
    # sort out interesting indices    
    track_lengths = np.array([ len(s) for s in x_emb_test ])
    fig, axes = make_evaluation_plot(y_true, y_pred, nn, track_lengths)
    
    plt.show()
    exit()
    
    output_dir = root_dir + "models/navigator/"
    method_string = "-recurrent"
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_string = "-emb" + str(embedding_dim)
    accuracy_string = "-acc" + str(int(round(next_neighbor_accuracy(y_true, y_pred, nn)*100)))
    output_filename = date_string + method_string + embedding_string + accuracy_string
    
    model.save(output_dir + output_filename)
    logging.info("Saved model to %s", output_dir+output_filename)
    
    fig.savefig(output_dir + output_filename + ".png")
    logging.info("Saved figure to %s", output_dir+output_filename+".png")    
    

if __name__ == "__main__":
    main(sys.argv)
