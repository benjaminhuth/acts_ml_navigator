import argparse
import logging
import json
import time
import pprint
import sys
import os
import collections

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from .config import *   
    


def parse_args_from_dictionary(dictionary):
    '''
    Reads in a dictionary, defines cmd args based on it and parses them.
    
    Parameters:
    * dictionary: A dictionary containing options
    
    Returns
    * dictionary: The modified dictionary
    '''
    
    # Basic parsing (why special handling of bool necessary?)
    parser = argparse.ArgumentParser()
    
    for key, value in dictionary.items():
        if type(value) == bool:
            parser.add_argument("--" + key, type=int)
        else:
            parser.add_argument("--" + key, type=type(value))
            
    parser.add_argument("--show_config", action='store_true')
        
    args = vars(parser.parse_args())
    
    if args['show_config'] == True:
        pprint.pprint(dictionary, width=2)
        exit(0)
        
    del args['show_config']
    
    num_args = 0
    for key, value in args.items():
        if value != None:
            num_args += 1
            arg_type = type(dictionary[key])
            dictionary[key] = arg_type(value)
            
    logging.info("Parsed %d command line arguments, run with the following params:", num_args)
    
    return dictionary



def init_options_and_logger(propagation_dir, output_dir, additional_options={}, log_level=logging.INFO, ):
    '''
    Init general options and the logger. Also reads options from command line.
    
    Parameters:
    * propagation_dir: directory to propagation data, usually config.get_navigation_training_dir(). If set to 'None', it is ignored.
    * output_dir: where to store the results
    * [OPT] additional_options (dictionary): User specified options
    * [OPT] log_level
    
    Returns:
    * dictionary with all options.
    '''
    
    # Init loggers
    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)
    
    options = get_common_options(output_dir)
    
    # Add additional_options
    for key, value in additional_options.items():
        options[key] = value
        
    # Set output dir
    options['output_dir'] = output_dir
        
    options = parse_args_from_dictionary(options)
    
    # Resolve 'propagation_file' from 'prop_data_size'
    if propagation_dir != None:
        assert os.path.exists(propagation_dir)
        propagation_files = extract_propagation_data(propagation_dir)
        
        if options['prop_data_size'] in propagation_files:
            options['propagation_file'] = propagation_files[ options['prop_data_size'] ]
        else:
            logging.error("Could not find a propagation dataset with %d Acts events", options['prop_data_size'])
            exit(1)
    
    # Print configuration
    pprint.pprint(options, width=2)
    sys.stdout.flush()
    
    # Tensorflow devices
    if options['disable_gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    tensorflow_devices = [ t for n, t in tf.config.list_physical_devices() ]
    logging.info("tensorflow devices: %s", tensorflow_devices)
    if not 'GPU' in tensorflow_devices and not options['disable_gpu']:
        logging.warning(">>> NO GPU AVAILABLE! <<<")
        
    return options


def extract_embedding_model(embedding_dir, embedding_dim):
    '''
    Returns path to the best embedding model. Assumes that model directories are named like '20201217-185433-emb10-acc38'.
    
    Parameters:
    * embedding_dir: Where to search for the models
    * embedding_dim: which dimension we search
    '''
    
    embedding_models = []
    
    EmbeddingInfo = collections.namedtuple("EmbeddingInfo", ["path", "bpsplit_z", "bpsplit_phi"])
    
    assert os.path.exists(embedding_dir)
    
    for entry in os.scandir(embedding_dir):        
        if entry.is_dir():
            parts = entry.name.split('-')
            
            if len(parts) != 6:
                continue
            
            dim = int(parts[2][3:])
            
            if dim != embedding_dim:
                continue
            
            #acc = int(parts[3][3:])
            bpsplit_z = int(parts[4][2:])
            bpsplit_phi = int(parts[5][2:])
            
            embedding_models.append( EmbeddingInfo(entry.path, bpsplit_z, bpsplit_phi) )
            
    if len(embedding_models) == 0:
        logging.error("Could not find a matching embedding model to dim %d in '%s'", embedding_dim, embedding_dir)
        exit(1)
    
    # filenames should be sorted from worst->best
    return embedding_models[-1]



def extract_propagation_data(propagation_dir):
    '''
    Function to get paths of propagation data
    
    Parameters:
    * propagation_dir: where to search for propagation *.csv files from Acts
    
    Returns:
    * dictionary { num_events : path }
    '''
    propagation_data = {}
    
    for entry in os.scandir(propagation_dir):
        if entry.is_file():
            filename_no_extension = entry.name[:-4]
            parts = filename_no_extension.split('-')
            
            if len(parts) != 4:
                continue
            
            num_events = int(parts[3][1:])
            propagation_data[num_events] = entry.path
    
    return propagation_data



def cantor_pairing(array):
    '''
    A bijective NxN -> N map, used for pair comparison
    
    Parameters:
    * array: A ndarray with shape (N,2), thus containing N pairs of numbers
    
    Retruns
    * ndarray of length N, containing unique values for each pair
    '''
    k0 = array[:,0]
    k1 = array[:,1]
    return 0.5 * (k0 + k1)*(k0 + k1 + 1) + k1




class RemainingTimeEstimator(tf.keras.callbacks.Callback):
    '''
    Keras callback which evaluates the remaining time by averaging over all previous epoch durations
    '''
    def __init__(self, num_epochs):
        self.times = np.zeros(num_epochs)
       
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        t1 = time.time()
        self.times[epoch] = t1 - self.t0
        
        remaining = np.mean(self.times[0:epoch+1]) * (len(self.times) - epoch)
        print("Estimate of remaining time: {}".format(time.strftime("%Hh:%Mm:%Ss", time.gmtime(remaining))))
        
        

def nn_index_matches_embedding_model(embedding_model, nn, num_tests=10):
    '''
    Test if the nn index gives the correct indices for a embedding model
    
    Parameters:
    * embedding_model: A keras model (which should consist only of a Embedding layer)
    * nn: A sklearn NN-index for the embedding
    * num_tests: How many random samples should be tested?
    
    Returns:
    * bool: does it match or not?
    '''
    
    assert nn.n_samples_fit_ == embedding_model.get_layer(index=1).get_config()['input_dim']
    
    ref_ids = np.random.randint(0, nn.n_samples_fit_, num_tests)
    embs = np.squeeze(embedding_model(ref_ids))
    
    test_ids = np.squeeze(nn.kneighbors(embs, 1, return_distance=False))
    
    return np.array_equal(test_ids, ref_ids)



def export_results(output_filename, model, figure, options):
    '''
    Exports all necessary things, if options['export'] is true.
    
    Parameters:
    * output_file: string, also containing the directory
    * model: the keras model
    * figure: matplotlib figure
    * options: the options dictionary
    '''
    
    assert os.path.exists(options['output_dir'])
    
    if not options['export']:
        logging.info("output filename would be '%s'",output_filename)
        return
        
    model.save(output_filename)
    logging.info("exported model to '%s'", output_filename)

    figure.savefig(output_filename + ".png")
    logging.info("exported chart to '%s'.png", output_filename)
    
    with open(output_filename + '.json','w') as conf_file:
        json.dump(options, conf_file, indent=4)
    logging.info("Exported configuration to '%s.json'", output_filename)
