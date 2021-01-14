import os
import sys
import time

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
        '#737373',  # Gray
        '#800000',  # Dark red
        '#ff3300',  # Bright red
        '#ffff00',  # Yellow
        '#00cc00',  # Light green
        '#006600'   # Dark green
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
    in the nearest (1,2,3,5,10,otherwise)-neigbors of the true value
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
        
        

def plot_detailed_neighbour_accuracy(log_epochs, neighbor_history):
    '''
    takes a list of epochs and a np.ndarray of the logged history
    returns an matplotlib (figure, axes) tuple
    '''
    assert neighbor_history.shape[0] == len(log_epochs) and neighbor_history.shape[1] == 6
    
    epochs = np.arange(0,len(neighbor_history)).astype(np.int)
    
    stacked_results = [
        np.zeros(len(neighbor_history)),
        np.sum(neighbor_history[:,5:6],axis=1),
        np.sum(neighbor_history[:,4:6],axis=1),
        np.sum(neighbor_history[:,3:6],axis=1),
        np.sum(neighbor_history[:,2:6],axis=1),
        np.sum(neighbor_history[:,1:6],axis=1),
        np.sum(neighbor_history[:,0:6],axis=1),
    ]
    
    colors = get_colors()
    
    fig, ax = plt.subplots()
    
    for i in range(neighbor_history.shape[1]):
        ax.fill_between(log_epochs,stacked_results[i],stacked_results[i+1],color=colors[i])
        ax.plot(log_epochs,stacked_results[i],color=colors[i])
        
    ax.set_xticks(log_epochs, minor=False)
    ax.legend(['not in best 10','in best 10','in best 5','in best 3','in best 2','correct'], loc='lower left', framealpha=1.)
    
    return fig, ax


###################################
# Track position based evaluation #
###################################

def compute_pos_list(track_lengths, num_edges):
    '''
    returns an array, containing integers which represent the position in the track of the ith element
    '''
    assert np.sum(track_lengths) == num_edges
    
    pos_list = []
    pos_in_track = 0
    n_track = 0
    
    for i in range(num_edges):
        if pos_in_track == track_lengths[n_track]:
            pos_in_track = 0
            n_track += 1
            
        pos_list.append(pos_in_track)
        pos_in_track += 1
    
    return np.array(pos_list)



def make_evaluation_plots(tracks_edges_start, tracks_edges_target, tracks_params, history, evaluate_edge, figsize=(16,10)):
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
    
    ##############
    # Evaluation #
    ##############
    
    max_track_length = max([ len(track) for track in tracks_edges_start ])
    
    score_matrix = pd.DataFrame(data = np.zeros((max_track_length, 8)),
                                    index=np.arange(max_track_length),
                                    columns=['in1','in2','in3','in5','in10','other','relative_score','num_edges'])
    
    times = []
        
    if 'TERM' in os.environ:
        print_progress_bar(0, len(tracks_edges_start))
    
    for i, (starts, targets, params) in enumerate(zip(tracks_edges_start, tracks_edges_target, tracks_params)):  
                
        t0 = time.time()
        
        for pos_in_track, (start, target, param) in enumerate(zip(starts, targets, params)):
            score_matrix = evaluate_edge(pos_in_track, start, target, param, score_matrix)
            
        t1 = time.time()
        
        times.append(t1 - t0)
        remaining = np.mean(np.array(times)) * (len(tracks_edges_start)-i)
        remaining_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(remaining))
            
        if not 'TERM' in os.environ:
            progress = 100*i/len(tracks_edges_start)
            print("Progress: {:.1f}% - estimated time remaining: {}".format(progress, remaining_str), flush=True)
        else:
            print_progress_bar(i+1, len(tracks_edges_start),
                               "Progress: ", "- estimated time remaining: {}".format(remaining_str))  
    
    # Normalize
    hits_per_pos = np.sum(score_matrix[['in1','in2','in3','in5','in10','other']].values, axis=1)
    total_num_edges = np.sum(score_matrix[['in1','in2','in3','in5','in10','other']].values.flatten())
    
    assert hits_per_pos.all() > 0
    assert total_num_edges > 0
    
    score_matrix['relative_score'] = score_matrix['relative_score'].to_numpy() / hits_per_pos
    score_matrix['num_edges'] = score_matrix['num_edges'].to_numpy() / hits_per_pos
    
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
    if 'accuracy' in history and 'val_accuracy' in history:
        ax[0,1].plot(history['accuracy'])
        ax[0,1].plot(history['val_accuracy'])
        ax[0,1].legend(["train accuracy", "validation accuracy"])
    else:
        ax[0,1].text(0.2,0.5,"no accuracy to plot")
    
    # Plot absolute distribution (in total)    
    for i, (name, series) in enumerate(score_matrix[['in1','in2','in3','in5','in10','other']].iteritems()):
        res = np.sum(series.to_numpy())
        norm_res = res / total_num_edges
        rects = ax[0,2].bar(i,norm_res,color=get_colors()[-(i+1)])
        autolabel(ax[0,2], rects)
        
    ax[0,2].set_title("Score (all edges)")
    ax[0,2].set_xlabel("score bins")
    ax[0,2].set_ylabel("absolute score")
    ax[0,2].set_ylim([0,1])
    ax[0,2].legend(score_matrix.columns.tolist())
    
    # Plot distribution per track pos
    current = np.zeros(max_track_length)
    for i, (name, series) in enumerate(score_matrix[['in1','in2','in3','in5','in10','other']].iteritems()):
        # dont plot 'other'
        if i == len(score_matrix.columns)-3:
            break
        
        current += series.to_numpy() / hits_per_pos
        ax[1,0].plot(current, color=get_colors()[-(i+1)])
        
    ax[1,0].set_title("Score (per track position)")
    ax[1,0].set_xlabel("track position")
    ax[1,0].set_ylabel("score")
    ax[1,0].set_ylim([0,1.1])
    ax[1,0].legend(score_matrix.columns.tolist()[:-1])
        
    # Plot relative score
    ax[1,1].set_title("Relative score (per track position)")
    ax[1,1].set_xlabel("track position")
    ax[1,1].set_ylabel("relative score (w.r.t. # edges)")
    ax[1,1].set_ylim([0,1.1])
    if not score_matrix['relative_score'].to_numpy().all() == 0:
        ax[1,1].plot(score_matrix['relative_score'].to_numpy())
    else:
        ax[1,1].text(0.2,0.5,"no relative_score to plot")
    
    # Plot possible edges per track pos
    ax[1,2].set_title("Mean graph edges per track position")
    ax[1,2].set_xlabel("track position")
    ax[1,2].set_ylabel("mean #edges")
    if not score_matrix['num_edges'].to_numpy().all() == 0:
        ax[1,2].plot(score_matrix['num_edges'].to_numpy())
    else:
        ax[1,2].text(0.2,0.5,"no relative_score to plot")
    
    return fig, ax, float(np.sum(score_matrix['in1'].to_numpy())/total_num_edges)
