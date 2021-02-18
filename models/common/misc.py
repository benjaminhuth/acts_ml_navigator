import argparse
import logging
import json
import time
import pprint
import sys
import os
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import pandas as pd

from .config import *   
    


def parse_args_from_dictionary(dictionary):
    '''
    Reads in a dictionary, defines cmd args based on it and parses them.
    
    Parameters:
    * dictionary: A dictionary containing options
    
    Returns
    * dictionary: The modified dictionary
    '''
    
    # Add arguments (TODO why special handling of bool necessary?)
    parser = argparse.ArgumentParser()
    
    for key, value in dictionary.items():
        if type(value) == bool:
            parser.add_argument("--" + key, type=int)
        else:
            parser.add_argument("--" + key, type=type(value))
            
    parser.add_argument("--show_config", action='store_true')
    parser.add_argument("--load_json", type=str)
     
    # Pars parameters
    args = vars(parser.parse_args())
    
    # Handle 'show_config'
    if args['show_config'] == True:
        pprint.pprint(dictionary, width=2)
        exit(0)
        
    del args['show_config']
    
    # Handle 'load_json'
    if args['load_json'] != None:
        with open(args['load_json'],'r') as conf_file:
            parsed = json.load(conf_file)
            
        for key, val in parsed.items():
            dictionary[key] = val
            
        logging.info("loaded configuration from '%s'", args['load_json'])
            
    del args['load_json']
    
    # Handle remaining args
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
        
    # Some assertions
    assert options['bpsplit_method'] == 'uniform' or options['bpsplit_method'] == 'density'
    assert os.path.exists(options['detector_file'])
    assert os.path.exists(options['output_dir'])
        
    # Return the processed options
    return options


def extract_embedding_model(embedding_dir, embedding_dim, bpsplit_z=None, bpsplit_phi=None, bp_method=None):
    '''
    Returns path to the best embedding model. Assumes that model directories are named like '20201217-185433-emb10-acc38'.
    
    Parameters:
    * embedding_dir (str): Where to search for the models
    * embedding_dim (int): which dimension we search
    * [OPT] bpsplit_z (int): If given, only accept with this beampipe split z
    * [OPT] bpsplit_ph (int)i: If given, only accept with this beampipe split phi
    
    Returns:
    * Named tuple EmbeddingInfo('path', 'bpsplit_z', 'bpsplit_phi') 
    '''
    
    embedding_models = []
    
    EmbeddingInfo = collections.namedtuple("EmbeddingInfo", ["path", "bpsplit_z", "bpsplit_phi"])
    
    assert os.path.exists(embedding_dir)
    
    for entry in os.scandir(embedding_dir):        
        if entry.is_dir():
            # Assert there exists a txt file for the z-split
            parts = entry.name.split('-')
            
            if len(parts) != 6:
                continue
            
            dim = int(parts[2][3:])
            
            if dim != embedding_dim:
                continue
            
            this_bpsplit_z = int(parts[4][2:-3])
            
            if bpsplit_z != None and bpsplit_z != this_bpsplit_z:
                continue
            
            this_bp_method = parts[4][-3:]
            
            if bp_method != None and this_bp_method != bp_method[0:3]:
                continue
            
            this_bpsplit_phi = int(parts[5][2:])
            
            if bpsplit_phi != None and bpsplit_phi != this_bpsplit_phi:
                continue
            
            bpsplit_z_file = os.path.join(embedding_dir, entry.name + ".txt")
            assert os.path.exists(bpsplit_z_file)
            
            bpsplit_z_bounds = np.loadtxt(bpsplit_z_file).tolist()
            
            assert len(bpsplit_z_bounds) >= 2
            assert len(bpsplit_z_bounds)-1 == this_bpsplit_z
            
            embedding_models.append( EmbeddingInfo(entry.path, bpsplit_z_bounds, this_bpsplit_phi) )
            
    if len(embedding_models) == 0:
        logging.error("Could not find a matching embedding model (dim=%d, bpz=%s, bpphi=%s, method=%s) in '%s'",
                      embedding_dim, bpsplit_z, bpsplit_phi, bp_method, embedding_dir)
        exit(1)
    
    # filenames should be sorted from worst->best
    return embedding_models[-1]



def extract_bpsplit_bounds(embedding_dir, z, phi, method):
    '''
    Finds a bpsplit_z file in the directory matching the criteria
    
    Parameters:
    * embedding_dir (str): Where to search for the models
    * z (int): z split
    * phi (int): phi split
    * method (str): 'uniform' or 'density'
    
    Returns:
    * bpsplit_z_bounds (list)
    '''
    search_str = "bz{}{}-bp{}".format(z, 'uni' if method == 'uniform' else 'den', phi)
    
    assert os.path.exists(embedding_dir)
    assert method == 'uniform' or method == 'density'
        
    for entry in os.scandir(embedding_dir):        
        filename = entry.name
        
        if not (".txt" in filename and search_str in filename):
            continue
        
        return np.loadtxt(entry.path).tolist()
    
    logging.error("could not find specified embedding (bpz=%d, bphi=%d, method=%s) in '%s'", z, phi, method, embedding_dir)
    exit()
        


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
        
        

def nn_index_matches_embedding_model(embedding_model, nn, num_tests=100, is_keras_model=True):
    '''
    Test if the nn index gives the correct indices for a embedding model
    
    Parameters:
    * embedding_model: A keras model (which should consist only of a Embedding layer)
    * nn: A sklearn NN-index for the embedding
    * num_tests: How many random samples should be tested?
    * is_keras_model: Is the model a keras model?
    
    Returns:
    * bool: does it match or not?
    '''
    
    if is_keras_model:
        assert nn.n_samples_fit_ == embedding_model.get_layer(index=1).get_config()['input_dim']
    
    ref_ids = np.random.randint(0, nn.n_samples_fit_, num_tests)
    embs = np.squeeze(embedding_model(ref_ids))
    
    test_ids = np.squeeze(nn.kneighbors(embs, 1, return_distance=False))
    
    return np.array_equal(test_ids, ref_ids)



def export_results(output_filename, model, figure, options):
    '''
    Exports all necessary things, if options['export'] is true.
    Also converts options['bpsplit_z'] from list back to int
    
    Parameters:
    * output_file: string, also containing the directory
    * model: the keras model
    * figure: matplotlib figure
    * options: the options dictionary
    '''
    
    assert os.path.exists(options['output_dir'])
    options['bpsplit_z'] = len(options['bpsplit_z'])-1
    
    if not options['export']:
        logging.info("output filename would be '%s'",output_filename)
        return
        
    model.save(output_filename)
    logging.info("exported model to '%s'", output_filename)

    figure.savefig(output_filename + ".png")
    logging.info("exported chart to '%s.png'", output_filename)
    
    with open(output_filename + '.json','w') as conf_file:
        json.dump(options, conf_file, indent=4)
    logging.info("Exported configuration to '%s.json'", output_filename)



def export_embedding_file(output_filename, detector_file, embedding_model, total_bp_split):
    '''
    
    '''
    
    geo_ids = pd.read_csv(detector_file, dtype={'geo_id': np.uint64})['geo_id'].to_numpy()[1:]
    total_node_num = len(geo_ids) + total_bp_split
    
    numbers = np.arange(total_node_num)
    embeddings = np.squeeze(embedding_model(numbers))
    
    embedding_dim = embeddings.shape[1]
    headers = [ "emb" + str(i) for i in range(embedding_dim) ]
    
    df = pd.DataFrame(data=embeddings, index=numbers, columns=headers)
    df.insert(loc=0, column='geoid', value=np.concatenate([np.arange(total_bp_split, dtype=np.uint64), geo_ids]))
    
    df.to_csv(output_filename)
    
    
    
