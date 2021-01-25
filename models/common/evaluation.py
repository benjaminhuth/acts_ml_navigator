import os
import sys
import time
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf



def get_colors():
    '''
    Returns the colors for accuracy in the order bad -> good (gray -> green)
    '''
    return [
        '#006600',  # Dark green
        '#00cc00',  # Light green
        '#ffff00',  # Yellow
        '#ff3300',  # Bright red
        '#800000',  # Dark red
        '#737373',  # Gray
    ]



def autolabel(ax, rects):
    """
    From: https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html
    
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
 
 
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    From: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



##################################
# Detailed neighborhood analysis #
##################################

def next_neighbor_accuracy(y_true, y_pred, nn):
    '''
    A custom accuracy with numpy and sklearn.NearestNeighbors. 
    Returns a float [0,1] reflecting how many of the prediction were nearest neighbors to the true values.
    Needs to be wraped in tf.py_function.
    '''
    assert len(y_true) == len(y_pred)
    
    y_pred_nn_idxs = nn.kneighbors(y_pred, 1, return_distance=False)
    y_true_idxs = nn.kneighbors(y_true, 1, return_distance=False)
    
    return np.sum(y_true_idxs == y_pred_nn_idxs)/len(y_true)


def neighbor_accuracy_detailed(y_true,y_pred, nn):
    '''
    Detailed neighborhood accuracy: Returns a list of floats [0,1] reflecting, how many predictions are
    in the nearest (1,2,3,5,10,otherwise)-neigbors of the true value.
    '''
    assert len(y_true) == len(y_pred)
    
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
    
    bad = len(y_true) - (in1 + in2 + in3 + in5 + in10)
    
    return np.array([in1, in2, in3, in5, in10, bad]) / len(y_true)



def evaluate_and_plot(tracks_edges_start, tracks_params, tracks_edges_target, history, evaluate_edge, figsize=(16,10)):
    '''
    Function that makes a collection of plots for evaluation of a model
    
    Parameters:
    * tracks_edges: a list of ndarrays with shape [track_length,2]
    * tracks_params: a list of ndarrays with shape [track_length,selected_params]
    * history: a history dictionary (must contain the key 'loss' and 'val_loss')
    * evaluate_edge: a callable to invoke to evaluate a specific edge
    * figsize: a tuple (width,height) in inches (?) for the figure
    
    Returns:
    * matploglib figure
    * axes array (containing 3x2 subplots)
    * score (ratio of overall 'in1' predictions)
    '''
    assert len(tracks_edges_start) == len(tracks_params) == len(tracks_edges_target)
    assert tracks_edges_start[0].shape == tracks_edges_target[0].shape
    assert tracks_params[0].shape[1] == 3 or tracks_params[0].shape[1] == 4
    
    ##############
    # Evaluation #
    ##############
    
    max_track_length = max([ len(track) for track in tracks_edges_start ])
    
    # Initialize the result
    EvaluationResult = collections.namedtuple("EvaluationResult", ["score_matrix", "beampipe_scores"])
    
    columns = ['in1','in2','in3','in5','in10','other','num_edges','num_edges_max','num_edges_min','relative_score']
    result = EvaluationResult(
        pd.DataFrame(data = np.zeros((max_track_length, len(columns))),
                     index=np.arange(max_track_length),
                     columns=columns),
        { 'in1': [], 'in2': [], 'in3': [], 'in5': [], 'in10': [], 'other': [] }
    )
    
    # Loop over all tracks
    times = []
        
    if 'TERM' in os.environ:
        print_progress_bar(0, len(tracks_edges_start))
    
    for i, (starts, targets, params) in enumerate(zip(tracks_edges_start, tracks_edges_target, tracks_params)):  
                
        t0 = time.time()
        
        for pos_in_track, (start, target, param) in enumerate(zip(starts, targets, params)):
            result = evaluate_edge(pos_in_track, start, target, param, result)
            
        t1 = time.time()
        
        times.append(t1 - t0)
        remaining = np.mean(np.array(times)) * (len(tracks_edges_start)-i)
        remaining_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(remaining))
            
        if not 'TERM' in os.environ:
            # Only print 10 progress statements
            if i % (len(tracks_edges_start) // 10) == 0:
                progress = 100*i/len(tracks_edges_start)
                print("Progress: {:.1f}% - estimated time remaining: {}".format(progress, remaining_str), 
                      flush=True)
        else:
            print_progress_bar(i+1, len(tracks_edges_start),
                               "Progress: ", "- estimated time remaining: {}".format(remaining_str))  
    
    # Normalize
    hits_per_pos = np.sum(result.score_matrix[['in1','in2','in3','in5','in10','other']].values, axis=1)
    total_num_edges = np.sum(result.score_matrix[['in1','in2','in3','in5','in10','other']].values.flatten())
    
    assert hits_per_pos.all() > 0
    assert total_num_edges > 0
    
    result.score_matrix['relative_score'] = result.score_matrix['relative_score'].to_numpy() / hits_per_pos
    result.score_matrix['num_edges'] = result.score_matrix['num_edges'].to_numpy() / hits_per_pos
    
    ############
    # Plotting #
    ############
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    
    # Plot loss
    ax[0,0].set_xlabel("epochs")
    ax[0,0].set_ylabel("loss")
    ax[0,0].set_title("Loss")
    if 'loss' in history and 'val_loss' in history:
        ax[0,0].plot(history['loss'])
        ax[0,0].plot(history['val_loss'])
        ax[0,0].legend(["train loss", "validation loss"])
    else:
        ax[0,0].text(0.2,0.5,"no loss to plot")
    
    # Plot accuracy
    ax[0,1].set_xlabel("epochs")
    ax[0,1].set_ylabel("accuracy")
    ax[0,1].set_title("Accuracy")
    
    accuracy_legend = []
    if 'accuracy' in history and 'val_accuracy' in history:
        ax[0,1].plot(history['accuracy'])
        ax[0,1].plot(history['val_accuracy'])
        accuracy_legend += ["train accuracy", "validation accuracy"]
    if 'binary_accuracy' in history and 'val_binary_accuracy' in history:
        ax[0,1].plot(history['binary_accuracy'])
        ax[0,1].plot(history['val_binary_accuracy'])
        accuracy_legend += ["train binary accuracy", "validation binary accuracy"]
    
    if len(accuracy_legend) == 0:
        ax[0,1].text(0.2,0.5,"no accuracy to plot")
    else:
        ax[0,1].legend(accuracy_legend)
    
    
    # Plot absolute distribution (in total)    
    for i, (name, series) in enumerate(result.score_matrix[['in1','in2','in3','in5','in10','other']].iteritems()):
        res = np.sum(series.to_numpy())
        norm_res = res / total_num_edges
        rects = ax[0,2].bar(i,norm_res,color=get_colors()[i])
        autolabel(ax[0,2], rects)
        
    ax[0,2].set_title("Score (all edges)")
    ax[0,2].set_xlabel("score bins")
    ax[0,2].set_ylabel("absolute score")
    ax[0,2].set_ylim([0,1])
    ax[0,2].legend(columns)
    
    # Plot distribution per track pos
    current = np.zeros(max_track_length)
    for i, (name, series) in enumerate(result.score_matrix[['in1','in2','in3','in5','in10','other']].iteritems()):
        # dont plot 'other'
        if columns[i] == 'other':
            break
        
        current += series.to_numpy() / hits_per_pos
        ax[1,0].plot(current, color=get_colors()[i])
        
    ax[1,0].set_title("Score (per track position)")
    ax[1,0].set_xlabel("track position")
    ax[1,0].set_ylabel("score")
    ax[1,0].set_ylim([0,1.1])
    ax[1,0].legend(columns)
        
    # Plot beampipe score historgram
    hist_data = np.array([ result.beampipe_scores[k] for k in result.beampipe_scores.keys() ], dtype=object)
    ax[1,1].set_title("Score at beampipe (wrt z coord)")
    ax[1,1].set_xlabel("z-coord bins")
    ax[1,1].set_ylabel("score")
    ax[1,1].hist(hist_data, 25, histtype='bar', stacked=True, color=get_colors())
    ax[1,1].legend(result.beampipe_scores.keys())
    
    # Plot possible edges per track pos
    ax[1,2].set_title("Mean graph edges per track position")
    ax[1,2].set_xlabel("track position")
    ax[1,2].set_ylabel("edges")
    if not result.score_matrix['num_edges'].to_numpy().all() == 0:
        ax[1,2].plot(result.score_matrix['num_edges'].to_numpy())
        ax[1,2].plot(result.score_matrix['num_edges_max'].to_numpy())
        ax[1,2].plot(result.score_matrix['num_edges_min'].to_numpy())
        ax[1,2].legend(['mean', 'max', 'min'])
    else:
        ax[1,2].text(0.2,0.5,"no relative_score to plot")
    
    return fig, ax, float(np.sum(result.score_matrix['in1'].to_numpy())/total_num_edges)




def fill_in_results(pos_in_track, score, surface_z_coord, result, num_targets=None):
    '''
    Fillst the entries of the EvaluationResult named tuple.
    
    Parameters:
    * pos_in_track: integer
    * score: in which of ['in1','in2','in3','in5','in10','other'] to put the result
    * surface_z_coord: z coord of the surface_z_coord
    * result: EvaluationResult
    * [OPT] num_targets: integer
    
    Returns:
    * EvaluationResult: modified result type
    '''
    
    if score == 0: 
        result.score_matrix.loc[pos_in_track, 'in1'] += 1
        if pos_in_track == 0: result.beampipe_scores['in1'].append( surface_z_coord )
    elif score == 1: 
        result.score_matrix.loc[pos_in_track, 'in2'] += 1
        if pos_in_track == 0: result.beampipe_scores['in2'].append( surface_z_coord )
    elif score == 2: 
        result.score_matrix.loc[pos_in_track, 'in3'] += 1
        if pos_in_track == 0: result.beampipe_scores['in3'].append( surface_z_coord )
    elif score < 5: 
        result.score_matrix.loc[pos_in_track, 'in5'] += 1
        if pos_in_track == 0: result.beampipe_scores['in5'].append( surface_z_coord )
    elif score < 10:
        result.score_matrix.loc[pos_in_track, 'in10'] += 1
        if pos_in_track == 0: result.beampipe_scores['in10'].append( surface_z_coord )
    else: 
        result.score_matrix.loc[pos_in_track, 'other'] += 1
        if pos_in_track == 0: result.beampipe_scores['other'].append( surface_z_coord )
    
    if num_targets != None:
        result.score_matrix.loc[pos_in_track, 'num_edges'] += num_targets
        result.score_matrix.loc[pos_in_track, 'relative_score'] += 1 - score/num_targets
        
        result.score_matrix.loc[pos_in_track, 'num_edges_max'] = \
            max(result.score_matrix.loc[pos_in_track, 'num_edges_max'], num_targets)
        result.score_matrix.loc[pos_in_track, 'num_edges_min'] = \
            min(result.score_matrix.loc[pos_in_track, 'num_edges_min'], num_targets)
    
    return result
