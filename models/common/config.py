import os

###################################
# Specifies all common parameters #
###################################


def get_root_dir():
    '''
    Returns the root data directory (TODO make configurable by cmake!)
    '''
    path = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    assert os.path.exists(path)
    return path


def get_navigation_training_dir(detector):
    '''
    Returns the directory where propagation logs for navigation training are saved
    '''
    assert detector == 'generic' or detector == 'itk'
    path = os.path.join(get_root_dir(), "detectors", detector, "navigation_training")
    assert os.path.exists(path)
    return path

def get_false_samples_dir(detector):
    '''
    Returns the directory where false sample for the pairwise scoring model are stored
    '''
    assert detector == 'generic' or detector == 'itk'
    path = os.path.join(get_root_dir(), "detectors", detector, "false_samples")
    assert os.path.exists(path)
    return path

def get_embedding_training_dir(detector):
    '''
    Returns the directory where propagation logs for embedding training are saved
    '''
    assert detector == 'generic' or detector == 'itk'
    path = os.path.join(get_root_dir(), "detectors", detector, "embedding_training")
    assert os.path.exists(path)
    return path


def get_detector_file(detector):
    '''
    Returns the .csv file where the list of detector surfaces is stored
    '''
    assert detector == 'generic' or detector == 'itk'
    path = os.path.join(get_root_dir(), "detectors", detector, "detector_surfaces.csv")
    assert os.path.exists(path)
    return path


def get_common_options(output_dir):
    '''
    In this functions the common options for all programs are specified.
    
    Parameters:
    * output_dir: where to store the results (must be relative to root_dir)
    
    Returns:
    * dictionary with all common options
    '''
    
    options = {
        'output_dir': output_dir,
        'disable_gpu': False,
        'export': True, 
        'show': True,
        'detector': 'generic',
        'use_real_space_as_embedding': False,
        'prop_data_size': 128,
        'bpsplit_z': 40,
        'bpsplit_phi': 1,
        'bpsplit_method': 'density',
        'epochs': 10,
        'embedding_dim': 10,
        'batch_size': 2048,
        'activation': 'relu',
        'learning_rate': 0.001,
        'test_split': 0.3,
        'validation_split': 0.3,
        'network_depth': 3,
        'layer_size': 500,
        'eval_smooth_rzmap': True,
        'eval_smooth_rzmap_radius': 35.0,
    }
    
    return options
