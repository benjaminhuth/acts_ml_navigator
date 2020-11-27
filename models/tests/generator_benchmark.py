import sys
sys.path.append("./../")

import datetime
import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

from timeit import Timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility.data_import import *


# A bijective NxN -> N map, used for pair comparison
def cantor_pairing(array):
    k0 = array[:,0]
    k1 = array[:,1]
    return 0.5 * (k0 + k1)*(k0 + k1 + 1) + k1


def main(argv):
    ######################
    # PREPARE GRAPH DATA #
    ######################
    
    nodes, geoid_edges, weights = generate_graph_from_data("../../data/logger/data-201104-120629.csv")
    
    logging.info("num nodes: %s",len(nodes))
    logging.info("num edges: %s",len(geoid_edges))
    
    # get all valid nodes
    detector_data = pd.read_csv("../../data/detector/detector_surfaces.csv", dtype={'geo_id': np.uint64})
    all_geo_ids = detector_data['geo_id'].to_numpy()
    all_numbers = detector_data['ordinal_id'].to_numpy()
    logging.info("detector coverage ratio: %s",len(nodes)/len(all_geo_ids))
    logging.info("surfaces missing: %s",len(all_geo_ids)-len(nodes))

    # Transform geoids to numbers starting from 0
    geoid_to_number = { geo_id : number for geo_id,number in zip(all_geo_ids, all_numbers) }

    connected_edges = np.empty(shape=geoid_edges.shape,dtype=np.int32)
    for number_edge, id_edge in zip(connected_edges,geoid_edges):
        number_edge[0] = geoid_to_number[id_edge[0]]
        number_edge[1] = geoid_to_number[id_edge[1]]
        
    edge_set = set([(e[0],e[1]) for e in connected_edges])
        
    #########################################
    # A vectorized version of the generator #
    #########################################
        
    def generate_batch_test_vec(batch_size):
        num_connected = batch_size//2
        num_unconnected = batch_size - num_connected
        
        # increase number of unconnecteds to account for later filtered out edges
        increase = 1.1
        
        losses = []
        iters = 1000
        iters_where_len_to_small = 0
        
        # connected edges
        idxs = np.arange(len(connected_edges))
        np.random.shuffle(idxs)
        
        x_connected = connected_edges[idxs[0:num_connected]].copy()
        y_connected = np.ones(num_connected)
        
        # unconnected edges
        x_unconnected = np.random.randint(0,len(all_geo_ids),(int(num_unconnected*increase),2))
        x_unconnected = np.unique(x_unconnected,axis=0)
        
        # TODO apply unique before to speed up?
        is_unconnected_mask = np.logical_not(np.isin(cantor_pairing(x_unconnected),cantor_pairing(connected_edges),assume_unique=True))
        is_no_loop_mask = x_unconnected[:,0] != x_unconnected[:,1]
        valid_mask = np.logical_and(is_unconnected_mask, is_no_loop_mask)
        
        x_unconnected = x_unconnected[valid_mask]
        y_unconnected = np.zeros(num_unconnected)


    ###############################################
    # A straight forward version of the generator #
    ###############################################
    
    def generate_batch_test_loop(batch_size):
        num_connected = batch_size//2
        num_unconnected = batch_size - num_connected
        
        # connected edges
        idxs = np.arange(len(connected_edges))
        np.random.shuffle(idxs)
        
        x_connected = connected_edges[idxs[0:num_connected]].copy()
        y_connected = np.ones(num_connected)
        
        # unconnected edges
        x_unconnected = np.random.randint(0,len(all_geo_ids),(num_unconnected,2))
        this_edge_set = edge_set.copy()
        
        for i in range(num_unconnected):
            while True:
                new_sample = np.random.randint(0,len(all_geo_ids),2)
        
                if new_sample[0] != new_sample[1] and (new_sample[0],new_sample[1]) not in this_edge_set:
                    x_unconnected[i] = new_sample
                    this_edge_set.add((new_sample[0],new_sample[1]))
                    break
                
        y_unconnected = np.zeros(num_unconnected)
        
    ######################
    # Benchmark and Plot #
    ######################
        
    vec_times = []
    loop_times = []
    batch_sizes = [ 128, 256, 512, 1024, 2048, 4096, 8192 ]
    
    for batch_size in batch_sizes:
        t_vec = Timer(lambda: generate_batch_test_vec(batch_size))
        t_loop = Timer(lambda: generate_batch_test_loop(batch_size))
        
        vec_times.append( t_vec.timeit(number=100) )
        loop_times.append( t_loop.timeit(number=100) )
        
        logging.info("timing for batch size %d - vec: %3f - loop: %3f",batch_size,vec_times[-1],loop_times[-1])
        
    plt.plot(batch_sizes, vec_times)
    plt.plot(batch_sizes, loop_times)
    plt.legend(["vec","loop"])
    plt.xscale("log", basex=2)
    plt.xticks(batch_sizes, batch_sizes)
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
