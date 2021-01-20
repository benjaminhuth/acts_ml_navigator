#
# Preprocessing pipeline:
#
# 1) Do beampipe splitting
# 2) GeoID to ordinal number (must respect changed ids from beampipe splitting
# 3) Categorize into tracks and go to numpy arrays
# 4) Train-test-split
#

import os
import logging
import time
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
    
    
    
def make_track_collection(start_ids, start_params, target_ids):
    '''
    Creats a named tuple that represents a collection of tracks
    '''
    TrackCollection = collections.namedtuple("TrackCollection", ["start_ids", "start_params", "target_ids"])
    return TrackCollection(start_ids, start_params, target_ids)

    
    
def beampipe_split(prop_data, z_split, phi_split, return_z_distribution=False):
    '''
    Maps all GeoIDs which are 0 to new numbers in [0, z_split * phi_split], dependent on their track parameters. 
    
    Parameters:
    * prop_data: pandas dataframe containing the propagation data
    * z_split: in how many parts the beampipe will be split in z-direction
    * phi_split: in how many parts the beampipe will be split angle wise
    * [OPT] return_z_distribution: if true, return z_distribution
    
    Returns:
    * pandas dataframe containing modified propagation data
    '''
    
    assert z_split > 0 and phi_split > 0
    assert np.amin( prop_data[ prop_data['start_id'] != 0 ]['start_id'].to_numpy() ) >= z_split*phi_split
    
    # Get z positions 
    bp_z_positions = prop_data[ prop_data['start_id'] == 0 ]['pos_z'].to_numpy()
    
    logging.debug("min_z = %f, max_z = %f", np.amin(bp_z_positions), np.amax(bp_z_positions))
    
    # Compute bin size. increase the z-range slightly to avoid border effects
    z_bin_size = ( np.amax(bp_z_positions) - np.amin(bp_z_positions) )*1.01 / z_split
    assert bp_z_positions.all() >= 0 and bp_z_positions.all() < np.amax(bp_z_positions)
    
    # Translate z coords to be all > 0, then divide by bin_size and cast to int
    bp_new_ids = ((bp_z_positions - np.amin(bp_z_positions)) / z_bin_size).astype(np.uint64)
    
    # For conditional return
    z_ids, z_ids_weights = np.unique(bp_new_ids, return_counts=True)
    
    # expand new ids so that there is space for phi-splitting
    bp_new_ids *= phi_split
    
    # then do phi-splitting
    bp_dir_x = prop_data[ prop_data['start_id'] == 0 ]['dir_x'].to_numpy()
    bp_dir_y = prop_data[ prop_data['start_id'] == 0 ]['dir_y'].to_numpy()
    
    # get angle and normalize to [0, 2pi]
    bp_phi_angles = np.arctan2(bp_dir_x, bp_dir_y) + np.pi
    assert bp_phi_angles.all() >= 0 and bp_phi_angles.all() < 2*np.pi
    
    phi_bin_size = 2*np.pi / phi_split
    
    bp_new_ids += (bp_phi_angles / phi_bin_size).astype(np.uint64)
    
    prop_data.loc[ prop_data['start_id'] == 0, 'start_id'] = bp_new_ids
    
    # Update z positions of the new splitted tracks
    # Use the fact, that ids with same z but different phi are next to each other
    new_z_positions = np.linspace(np.amin(bp_z_positions), np.amax(bp_z_positions), z_split, endpoint=False) + z_bin_size/2
    low_z_bounds = np.arange(0, phi_split*z_split, phi_split)
    high_z_bounds = low_z_bounds + phi_split
    assert len(new_z_positions) == len(low_z_bounds) == len(high_z_bounds)
    
    for low, high, new_pos in zip(low_z_bounds, high_z_bounds, new_z_positions):
        logging.debug("the ids %s get new z coord %f", np.arange(low, high), new_pos)
        prop_data.loc[ (prop_data['start_id'] >= low) & (prop_data['start_id'] < high), 'start_z' ] = new_pos
    
    if not return_z_distribution:
        return prop_data
    else:
        return prop_data, (z_ids, z_ids_weights)
    


def categorize_into_tracks(prop_data, total_beampipe_split, selected_params):
    '''
    Splits the list of connections into tracks, assuming the beampipe is represented by a GeoID which is smaller than total_beampipe_split (so beampipe_splitting can be done before)
    
    Parameters:
    * prop_data: pandas dataframe containing the propagation data
    * total_beampipe_split: in how many parts the beampipe was split (positive integer)
    * selected_params: which track parameters are extracted (list of strings, e.g. ['dir_x', 'dir_y', 'dir_z'])
    
    Returns:
    * 3 lists of ndarrays: x_ids, y_ids, x_params
    '''    
    assert total_beampipe_split > 0
    
    sep_idxs = prop_data[prop_data['start_id'] < total_beampipe_split].index.to_numpy()
    sequence_lengths = np.diff(sep_idxs)

    x_geoids = prop_data['start_id'].to_numpy()
    y_geoids = prop_data['end_id'].to_numpy()
    x_params = prop_data[selected_params].to_numpy().astype(np.float32)
    
    x_tracks_geoids = []
    y_tracks_geoids = []
    x_tracks_params = []
    
    for i in range(len(sep_idxs)-1):
        x_tracks_geoids.append(x_geoids[sep_idxs[i]:sep_idxs[i+1]])
        y_tracks_geoids.append(y_geoids[sep_idxs[i]:sep_idxs[i+1]])
        x_tracks_params.append(x_params[sep_idxs[i]:sep_idxs[i+1]])
        
    x_tracks_geoids.append(x_geoids[sep_idxs[-1]:])
    y_tracks_geoids.append(y_geoids[sep_idxs[-1]:])
    x_tracks_params.append(x_params[sep_idxs[-1]:])
    
    assert len(x_tracks_geoids) == len(y_tracks_geoids) == len(x_tracks_params)
    
    return make_track_collection(x_tracks_geoids, x_tracks_params, y_tracks_geoids)
    


def geoid_to_ordinal_number(prop_data, detector_data, total_beampipe_split):
    '''
    Maps the Acts GeoID to an ordinal number [0, num_nodes - 1 + total_beampipe_split]. 
    
    Parameters:
    * prop_data: pandas dataframe containing propagation data
    * total_beampipe_split: in how many parts the beampipe was split (positive integer)
    * detector_data: pandas dataframe containing a mapping between geoids and ordinal numbers
    
    Returns:
    * pandas dataframe containing modified propagation data
    '''
    
    # get all geoids except beampipe
    all_geo_ids = detector_data['geo_id'].to_numpy()[1:]
    
    # get all ordinal numbers except beampipe
    all_numbers = detector_data['ordinal_id'].to_numpy()[1:]
    
    # shift all numbers to account for beampipe split
    all_numbers = all_numbers - 1 + total_beampipe_split
    
    # controll if everything worked
    assert np.amin(all_numbers) == total_beampipe_split
    
    # make dictionary
    geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }
    
    # add beampipe values (here we have an identity mapping)
    for n in range(total_beampipe_split):
        geoid_to_number[n] = n
    
    # use numpy to do this, because in pandas it is awful slow
    ids = prop_data[['start_id','end_id']].to_numpy()
    
    for i in range(len(ids)):
        ids[i,0] = geoid_to_number[ ids[i,0] ]
        ids[i,1] = geoid_to_number[ ids[i,1] ]
        
    prop_data[['start_id','end_id']] = ids
        
    return prop_data



def split_in_train_and_test_set(track_collection, test_split):
    '''
    Splits the tracks into two sets of train and test data.
    
    Parameters:
    * track_collection: of the named tuple type TrackCollection
    * test_split: float between 0 and 1
    
    Returns
    * 2 TrackCollection
    '''
    assert type(track_collection).__name__ == 'TrackCollection'
    assert test_split > 0 and test_split < 1
    
    train_starts, test_starts, train_params, test_params, train_targets, test_targets = \
        train_test_split(track_collection.start_ids, track_collection.start_params, 
                         track_collection.target_ids, test_size=test_split)
    
    return make_track_collection(train_starts, train_params, train_targets), \
           make_track_collection(test_starts, test_params, test_targets)
    
    

def make_graph_map(prop_data, total_node_num=None):
    '''
    Generates a dictionary { id : ndarray of ids } which represents the graph.
    
    Parameters:
    * prop_data (pandas dataframe)
    * OPT: total_node_num (int), enables some logs
    
    Returns:
    * dictionary { start : (ndarray of targets, ndarray of weights, real space position) }
    '''
    
    # Collect surface positions   
    surface_position_map = { }
    
    for column in ['start_id', 'end_id']:
        positions = prop_data[['start_x', 'start_y', 'start_z']].to_numpy()
        surfaces = prop_data[column].to_numpy()
        
        _, idxs = np.unique(surfaces, return_index=True)
        
        for idx in idxs:
            surface_position_map[ surfaces[idx] ] = positions[idx]
            
    # Find edges and weights
    graph_edges = prop_data[['start_id', 'end_id']].to_numpy()
    graph_edges, weights = np.unique(graph_edges, axis=0, return_counts=True)
    
    graph_nodes = np.unique(graph_edges)
    assert len(graph_nodes) <= total_node_num
    
    GraphNode = collections.namedtuple('GraphNode', ['targets', 'weights', 'position'])
    
    def targets_and_weights(start):
        mask = np.equal(graph_edges[:,0],start)
        return GraphNode( graph_edges[:,1][mask], weights[mask], surface_position_map[start] )
    
    graph_map = { start: targets_and_weights(start) for start in graph_nodes }
    
    # Do some logging
    if total_node_num != None:        
        logging.info("graph info: num surfaces = %d, num edges = %d, max weight = %d, min weight = %d", 
                    len(graph_nodes), len(graph_edges), np.amax(weights), np.amin(weights))
        logging.info("detector coverage of imported graph: %.2f%%, %d surfaces missing",
                    (len(graph_nodes)/total_node_num)*100, total_node_num - len(graph_nodes))
    
    return graph_map

            
    


############################
# Test preprocessing chain #
############################

if __name__ == "__main__":
    print("Test preprocessing chain...", flush=True)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.DEBUG)
    
    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    propagation_file = root_dir + "logger/navigation_training/data-201217-162645-n1.csv"
    detector_file = root_dir + "detector/detector_surfaces.csv"
    
    prop_data = pd.read_csv(propagation_file, dtype={'start_id': np.uint64, 'end_id': np.uint64})
    detector_data = pd.read_csv(detector_file, dtype={'geo_id': np.uint64})    
    
    
    ## Step 1: Beampipe split
    phi_split = 4
    z_split = 3
    step_1_res = beampipe_split(prop_data.copy(), z_split, phi_split)

    # NOTE this only works at small split sizes, so all are bins are filled
    initial_num_unique_ids = len(np.unique(prop_data[['start_id','end_id']].to_numpy()))
    step_1_num_unique_ids = len(np.unique(step_1_res[['start_id','end_id']].to_numpy()))
    
    assert initial_num_unique_ids - 1 + z_split*phi_split == step_1_num_unique_ids
    print("[ OK ] beampipe_split(...)", flush=True)
    
    
    ## Step 2: GeoID -> Number
    step_2_res = geoid_to_ordinal_number(step_1_res.copy(), detector_data, z_split*phi_split)
    step_2_unique_ids = np.unique(step_1_res[['start_id','end_id']].to_numpy())
    
    assert step_2_unique_ids.all() < len(step_2_unique_ids)
    print("[ OK ] geoid_to_ordinal_number(...)", flush=True)
    
    
    ## Step 3: Categorize into tracks    
    selected_params = ['dir_x','dir_y','dir_z']
    x_numbers, x_params, y_numbers = categorize_into_tracks(step_2_res.copy(), z_split*phi_split, selected_params)
    
    assert len(x_numbers) == len(x_params) == len(y_numbers)
    
    initial_sep_idxs = prop_data[prop_data['start_id'] == 0].index.to_numpy()
    track_lengths_ref = np.array([len(track) for track in x_numbers])
    track_lengths_check = np.append(np.diff(initial_sep_idxs), len(x_numbers[-1]))
    assert np.equal(track_lengths_ref, track_lengths_check).all()
    
    print("[ OK ] categorize_into_tracks(...)", flush=True)
