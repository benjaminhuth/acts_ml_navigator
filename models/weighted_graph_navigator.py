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



def evaluate_edge(pos, start, target, param, score_matrix, graph_edge_map):
    '''
    Function which evaluates all metrics for a single edge in a track.
    It uses no model, but only the weight of an edge as a criterion
    
    Parameters:
    * pos: the position in the track (int)
    * start: a id representing the start surface
    * target: a id representing the target surface
    * param: <unused>
    * score_matrix: a pandas df to fill in the results
    * model: the model to predict the score of a tuple [start_surface, end_surface, params]
    * graph_edge_map: a dictionary { start: namedtuple( targets, weights ) }
    
    Returns:
    * modified score_matrix (pandas dataframe)
    '''
    
    targets = graph_edge_map[ start ].targets
    weights = graph_edge_map[ start ].weights
    
    # sort targets by weights
    idxs = np.flip(np.argsort(weights))
    targets = targets[idxs]
    
    # Find where in the list the correct result is
    correct_pos = int(np.argwhere(np.equal(targets,target)))
    
    # Fill abs_score_matrix
    if correct_pos == 0: score_matrix.loc[pos, 'in1'] += 1
    elif correct_pos == 1: score_matrix.loc[pos, 'in2'] += 1
    elif correct_pos == 2: score_matrix.loc[pos, 'in3'] += 1
    elif correct_pos < 5: score_matrix.loc[pos, 'in5'] += 1
    elif correct_pos < 10: score_matrix.loc[pos, 'in10'] += 1
    else: score_matrix.loc[pos, 'other'] += 1
    
    # Fill res_scores (do the 1- to let the best result be 1)
    score_matrix.loc[pos, 'relative_score'] += 1 - correct_pos/len(targets)
    score_matrix.loc[pos, 'num_edges'] += len(targets)
    
    return score_matrix



#####################
# The main function #
#####################

def main():
    options = init_options_and_logger(get_navigation_training_dir(),
                                      os.path.join(get_root_dir(), "models/weighted_graph_navigator/"))   
    
    embedding_dir = os.path.join(get_root_dir(), 'models/target_pred_navigator/embeddings/')
    options['embedding_file'] = extract_embedding_model(embedding_dir, options['embedding_dim'])

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
    tracks_start, track_params, tracks_end = categorize_into_tracks(prop_data, total_node_num, ['qop'])
    
    # Make graph
    graph_edge_map = generate_graph_edge_map(prop_data, total_node_num)
    
    logging.info("Imported %d tracks, the maximum sequence length is %d",
                 len(tracks_start), max([ len(track) for track in tracks_start ]))


    ##############
    # Evaluation #
    ##############
    
    fig, axes, score = make_evaluation_plots(tracks_start, tracks_end, track_params, {},
                                             lambda a,b,c,d,e: evaluate_edge(a,b,c,d,e,graph_edge_map))
    
    bpsplit_str = "bp split: z={}, phi={}".format(options['beampipe_split_z'],options['beampipe_split_phi'])
    fig.suptitle("Weighted Graph Nav: {}".format(bpsplit_str), fontweight='bold')
    fig.text(0,0,"Training data: " + options['propagation_file'])
    
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
