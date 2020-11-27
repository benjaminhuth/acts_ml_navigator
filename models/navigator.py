import os
import sys
import datetime
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_addons as tfa
tf.get_logger().setLevel('ERROR')

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from utility.data_import import *

def main(argv):
    ########################
    # Import detector data #
    ########################
    
    #model_dir = '../data/embeddings/' + get_sorted_model_dirs('../data/embeddings/')[0]
    model_dir = '../data/embeddings/20201126-163733-emb50-acc40-dot'

    # Load model
    embedding_encoder = tf.keras.models.load_model(model_dir, compile=False)
    logging.info("loaded model from '%s'",model_dir)
    
    embedding_shape = embedding_encoder(0).shape
    logging.info("embedding shape: %s",embedding_shape)
    
    # Get all valid nodes
    detector_filename = "../data/detector/detector_surfaces.csv"
    detector_data = pd.read_csv(detector_filename, dtype={'geo_id': np.uint64})
    all_geo_ids = detector_data['geo_id'].to_numpy()
    all_numbers = detector_data['ordinal_id'].to_numpy()

    # Transformation dictionary
    geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }
    logging.info("loaded detector data from '%s'", detector_filename)
    
    # Establish neighbouring index
    all_embeddings = np.squeeze(embedding_encoder(all_numbers).numpy())
    nn = NearestNeighbors()
    nn.fit(all_embeddings)
    logging.info("built neighbouring index")
    
    ##########################
    # Generate training data #
    ##########################
    
    # Import data
    propagation_file = "../data/logger/navigation_training/data-201113-190908-n8.csv"
    propagation_data = pd.read_csv(propagation_file, dtype={'start_id': np.uint64, 'end_id': np.uint64})
    logging.info("loaded propagation data from '%s', containing training %d samples",propagation_file,len(propagation_data.index))

    # Transformation dictionary
    geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }
    
    # Extract data
    x_train_ids = propagation_data['start_id'].to_numpy()
    x_train_params = propagation_data[['pos_x','pos_y','pos_z','dir_x','dir_y','dir_z','qop']].to_numpy()
    y_train_ids = propagation_data['end_id'].to_numpy()
    logging.info("extracted training data")
    
    # Transform data
    x_train_embs = np.array([ geoid_to_number[geoid] for geoid in x_train_ids ])
    x_train_embs = np.squeeze(embedding_encoder(x_train_embs))
    
    y_train_embs = np.array([ geoid_to_number[geoid] for geoid in y_train_ids ])
    y_train_embs = np.squeeze(embedding_encoder(y_train_embs))
    
    # check compatibility
    assert len(x_train_embs) == len(x_train_params) == len(y_train_embs)
    
    # some printing
    logging.info("processed training data")    
    logging.info("x_train_embs.shape: %s",x_train_embs.shape)
    logging.info("x_train_params.shape: %s",x_train_params.shape)
    logging.info("y_train_embs.shape: %s",y_train_embs.shape)
    
    # Train test split (includes shuffle)
    test_split = 0.1
    x_train_embs, x_test_embs, x_train_params, x_test_params, y_train_embs, y_test_embs = \
        train_test_split(x_train_embs, x_train_params, y_train_embs, test_size=test_split)
    logging.info("test split: %.2f - samples train vs. test: %d - %d",test_split,len(x_train_embs),len(x_test_embs))
    
    ##############################
    # Build the navigation model #
    ##############################
    
    # Implementation of custom accuracy with numpy and sklearn
    def neighbor_accuracy_impl_numpy(y_true, y_pred):
        y_pred_nn_idxs = nn.kneighbors(y_pred.numpy(), 1, return_distance=False)
        y_true_idxs = nn.kneighbors(y_true.numpy(), 1, return_distance=False)
        
        return np.sum(y_true_idxs == y_pred_nn_idxs)/len(y_true)
        
    def neighbor_accuracy(y_true, y_pred):
        return tf.py_function(neighbor_accuracy_impl_numpy, inp=[y_true,y_pred], Tout=tf.float32)
        #return neighbor_accuracy_impl_tf(y_true,y_pred)
    
    # The navigation model
    def build_navigation_model(embedding_dim,hidden_layers,learning_rate):
        assert len(hidden_layers) >= 1
        
        input_params = tf.keras.Input(4)
        input_id = tf.keras.Input(embedding_dim)
        
        a = tf.keras.layers.Concatenate()([input_params,input_id])
        
        a = tf.keras.layers.Dense(hidden_layers[0])(a)
        a = tf.keras.layers.Activation(tf.nn.relu)(a)
        
        for i in range(1,len(hidden_layers)):
            a = tf.keras.layers.Dense(hidden_layers[i])(a)
            a = tf.keras.layers.Activation(tf.nn.relu)(a)
        
        output_id = tf.keras.layers.Dense(embedding_dim)(a)
        
        model = tf.keras.Model(inputs=[input_id,input_params],outputs=[output_id])
        #model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.MeanSquaredError(),
            #metrics=[neighbor_accuracy]
        )
        
        return model
    
    model_params = {
        'embedding_dim': embedding_shape[0],
        'hidden_layers': [500,500,500,500],
        'learning_rate': 0.001
    }
    
    navigation_model = build_navigation_model(**model_params)
    
    ###################
    # Do the training #
    ###################
    
    def neighbor_accuracy_detailed(y_true,y_pred):
        in1 = in2 = in3 = in5 = in10 = bad = 0
        
        y_true_idx = nn.kneighbors(y_true, 1, return_distance=False).flatten()
        y_pred_nearest_idxs = nn.kneighbors(y_pred, 10, return_distance=False)
        
        in1 += np.sum(y_true_idx == y_pred_nearest_idxs[:,0])
        in2 += np.sum(y_true_idx == y_pred_nearest_idxs[:,1])
        in3 += np.sum(y_true_idx == y_pred_nearest_idxs[:,2])
        in5 += np.sum(np.logical_or(
            y_true_idx == y_pred_nearest_idxs[:,3],
            y_true_idx == y_pred_nearest_idxs[:,4]
        ))
        in10 += np.sum(np.logical_or.reduce((
            y_true_idx == y_pred_nearest_idxs[:,5],
            y_true_idx == y_pred_nearest_idxs[:,6],
            y_true_idx == y_pred_nearest_idxs[:,7],
            y_true_idx == y_pred_nearest_idxs[:,8],
            y_true_idx == y_pred_nearest_idxs[:,9]
        )))
        
        bad = len(y_test_embs) - (in1 + in2 + in3 + in5 + in10)
        
        return [in1, in2, in3, in5, in10, bad]
    
    class NeighborDistributionCallback(tf.keras.callbacks.Callback):
        def compute_accuray(self):
            y_true = y_test_embs
            y_pred = self.model.predict([x_test_embs, x_test_params[:,3:7]])
            return neighbor_accuracy_detailed(y_true,y_pred)
        
        def on_train_begin(self, logs=None):
            self.results = []
            self.results.append(self.compute_accuray())
            
        def on_epoch_end(self, epoch, logs=None): 
            if epoch % 10 == 0:
                self.results.append(self.compute_accuray())
            
        def on_train_end(self, logs=None):
            self.model.history.history['neighbor_accuracy_detailed'] = self.results
            
    
    logging.info("start training")
    history = navigation_model.fit(
        x=[x_train_embs, x_train_params[:,3:7]], 
        y=y_train_embs,
        validation_data=([x_test_embs, x_test_params[:,3:7]], y_test_embs),
        batch_size=256,
        epochs=50,
        verbose=2,
        callbacks=[NeighborDistributionCallback()]
    )
    
    ############################
    # Some advanced validation #
    ############################
    
    y_test_pred = navigation_model([x_test_embs, x_test_params[:,3:7]])
    
    final_distribution = neighbor_accuracy_detailed(y_test_embs,y_test_pred.numpy())
    final_distribution_perc = np.array(final_distribution)/len(y_test_embs)
    logging.info("Distribution of test results (in best 1,2,3,5,10,otherwise):")
    logging.info("absolute: %d\t%d\t%d\t%d\t%d\t%d",*final_distribution)
    logging.info("percent:  %.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f",*final_distribution_perc)
    
    # comparison to neighbor_accuracy_impl_numpy
    assert neighbor_accuracy_impl_numpy(tf.constant(y_test_embs), y_test_pred) == final_distribution_perc[0]
    
    # plot
    neighbor_history = np.array(history.history['neighbor_accuracy_detailed'])  
    epochs = np.arange(0,len(neighbor_history)).astype(np.int)
    
    stacked_results = [
        np.zeros(len(neighbor_history)),
        np.sum(neighbor_history[:,5:6],axis=1),
        np.sum(neighbor_history[:,4:6],axis=1),
        np.sum(neighbor_history[:,3:6],axis=1),
        np.sum(neighbor_history[:,2:6],axis=1),
        np.sum(neighbor_history[:,1:6],axis=1),
        np.sum(neighbor_history[:,0:6],axis=1),
    ]
    
    colors = [
        '#737373',
        '#800000',
        '#ff3300',
        '#ffff00',
        '#00cc00',
        '#006600'
    ]
    
    for i in range(neighbor_history.shape[1]):
        plt.fill_between(epochs,stacked_results[i],stacked_results[i+1],color=colors[i])
        plt.plot(epochs,stacked_results[i],color=colors[i])
        
    plt.xticks(epochs)
    plt.legend(['not in best 10','in best 10','in best 5','in best 3','in best 2','correct'], loc='lower left', framealpha=1.)
    plt.show()    
        
    # export
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    accuracy_string = "-acc" + str(int(round(final_distribution_perc[0]*100)))
    output_filename = "../data/models/navigator/" + date_string + accuracy_string
        
    navigation_model.save(output_filename)
    logging.info("exported model to '" + output_filename + "'")
    
    
if __name__ == "__main__":
    main(sys.argv)
