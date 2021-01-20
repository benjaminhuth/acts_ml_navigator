import os
import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.preprocessing import *
from common.evaluation import *
from common.misc import *
from common.plot_embedding import *



def evaluate_edge(pos, start, target, param, result, graph_map):
    '''
    Function which evaluates all metrics for a single edge in a track.
    It uses no model, but only the weight of an edge as a criterion
    
    Parameters:
    * pos: the position in the track (int)
    * start: a id representing the start surface
    * target: a id representing the target surface
    * param: <unused>
    * result: named tuple EvaluationResult
    * graph_map: a dictionary { start: namedtuple( targets, weights, real space position ) }
    
    Returns:
    * modified score_matrix (pandas dataframe)
    '''
    node_data = graph_map[ start ]
    
    targets = node_data.targets
    weights = node_data.weights
    
    # sort targets by weights
    idxs = np.flip(np.argsort(weights))
    targets = targets[idxs]
    
    # Find where in the list the correct result is
    correct_pos = int(np.argwhere(np.equal(targets,target)))
    
    # Fill abs_score_matrix
    return fill_in_results(pos, correct_pos, node_data.position[2], result,  len(targets))



#####################
# The main function #
#####################

def main():
    options = init_options_and_logger(get_navigation_training_dir(),
                                      os.path.join(get_root_dir(), "models/weighted_graph_navigator/"))   

    ###############
    # Import data #
    ###############
    
    detector_data = pd.read_csv(options['detector_file'], dtype={'geo_id': np.uint64})
    prop_data = pd.read_csv(options['propagation_file'], dtype={'start_id': np.uint64, 'end_id': np.uint64})
    
    total_beampipe_split = options['beampipe_split_z']*options['beampipe_split_phi']
    total_node_num = len(detector_data.index) - 1 + total_beampipe_split
    
    # Beampipe split and new mapping
    prop_data = beampipe_split(prop_data, options['beampipe_split_z'], options['beampipe_split_phi'])
    prop_data = geoid_to_ordinal_number(prop_data, detector_data, total_node_num)
    
    # Categorize into tracks (also needed for testing later)
    tracks_start, track_params, tracks_end = categorize_into_tracks(prop_data, total_node_num, ['dir_x', 'dir_y', 'dir_z'])
    
    # Make graph
    graph_edge_map = make_graph_map(prop_data, total_node_num)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(tracks_start), max([ len(track) for track in tracks_start ]))


    ##############
    # Evaluation #
    ##############
    
    fig, axes, score = evaluate_and_plot(tracks_start, track_params, tracks_end, {},
                                         lambda a,b,c,d,e: evaluate_edge(a,b,c,d,e,graph_edge_map))
    
    bpsplit_str = "bp split: z={}, phi={}".format(options['beampipe_split_z'],options['beampipe_split_phi'])
    fig.suptitle("Weighted Graph Nav: {}".format(bpsplit_str), fontweight='bold')
    fig.text(0,0,"Training data: " + options['propagation_file'])
    
    if options['show']:
        plt.show()
           
    ##########
    # Export #
    ##########
           
    date_str   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    size_str   = "-n{}".format(options['prop_data_size'])
    acc_str    = "-acc{}".format(round(score*100))
    output_filename = date_str + size_str + acc_str

    if options['export']:
        fig.savefig(options['output_dir'] + output_filename + ".png")
        logging.info("exported chart to '" + options['output_dir'] + output_filename + ".png" + "'")
    else:
        logging.info("output filename would be: '%s'",output_filename)



if __name__ == "__main__":
    main()
