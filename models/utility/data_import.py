import os
import numpy as np
import pandas as pd


def get_sorted_model_dirs(base_dir):
    '''
    Returns subdirectories sorted by accuracy (best to worst). 
    Assumes that the subdirectories are named like `20201102-192109-emb10-acc100`
    '''
    model_dirs = np.array(next(os.walk(base_dir))[1])
    idxs = np.argsort(np.array([ subdir[subdir.find('acc')+3:] for subdir in model_dirs ]).astype(int))
    return np.flip(model_dirs[idxs])


def generate_graph_from_data(filename, return_pos_and_dir=False):
    '''
    returns nodes, edges, weights
    '''
    graph_data = pd.read_csv(filename, dtype={'start_id': np.uint64, 'end_id': np.uint64})

    edges = graph_data[["start_id","end_id"]].to_numpy()
    
    nodes = np.unique(edges.flatten())

    # transform doublets in edges to weights
    edges, idxs, weights = np.unique(edges,axis=0,return_index=True, return_counts=True)
    
    if return_pos_and_dir:
        dirs = graph_data[['dir_x','dir_y','dir_z']].to_numpy()
        dirs = dirs[idxs]
        
        poss = graph_data[['pos_x','pos_y','pos_z']].to_numpy()
        poss = poss[idxs]
        
        assert len(dirs) == len(edges) == len(poss)
        
        return nodes, edges, weights, poss, dirs
        
    
    return nodes, edges, weights


def get_list_of_all_geoids(detector_file):    
    all_nodes = pd.read_csv(detector_file,dtype={'geometry_id': np.uint64})['geometry_id'].to_list()
    return [0] + all_nodes # add beampipe
