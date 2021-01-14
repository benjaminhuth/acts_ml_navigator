import argparse
import logging
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
        
    args = vars(parser.parse_args())
    
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
    * propagation_dir: directory to propagation data, usually config.get_navigation_training_dir()
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
    returns list of tuples (path, num_simulated_acts_events)
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
    '''
    k0 = array[:,0]
    k1 = array[:,1]
    return 0.5 * (k0 + k1)*(k0 + k1 + 1) + k1




class RemainingTimeEstimator(tf.keras.callbacks.Callback):
    def __init__(self, num_epochs):
        self.times = []
        self.num_epochs = num_epochs
       
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        t1 = time.time()
        self.times.append(t1-self.t0)
        
        remaining = np.mean(np.array(self.times)) * (self.num_epochs - epoch)
        print("Estimate of remaining time: {}".format(time.strftime("%Hh:%Mm:%Ss", time.gmtime(remaining))))
        
        
