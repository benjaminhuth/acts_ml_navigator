import os
import sys
import time
import collections
import pprint
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors



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


def get_score_categories():
    '''
    Returns a list with the score categoires
    '''
    return ['correct ','in best 2','in best 3','in best 5','in best 10','worse']



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
    if not 'TERM' in os.environ or iteration > total:
        return
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        

class ProgressBar:
    '''
    Class wrapping the print_progress_bar function with time estimate functionality
    '''
    def __init__(self, total, prefix = "Progress:"):
        self.total = total
        self.prefix = prefix
        self.t0 = time.time()
        self.total_time = 0
        self.i = 0
        
        print_progress_bar(self.i, self.total, self.prefix)
        
        
    def print_bar(self):
        t1 = time.time()
        
        self.total_time += (t1 - self.t0)
        self.t0 = t1
        
        self.i += 1
        remaining = (self.total_time / self.i) * (self.total - self.i)        
        remaining_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(remaining))
        
        print_progress_bar(self.i, self.total, self.prefix, "- estimated time remaining: {}".format(remaining_str))



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



def add_subplot_zoom(figure):
    '''
    Allows to shift+click subplot to zoom on it.
    Source: https://stackoverflow.com/questions/44997029/matplotlib-show-single-graph-out-of-object-of-subplots
    '''
    zoomed_axes = [None]
    def on_click(event):
        ax = event.inaxes

        if ax is None:
            # occurs when a region not in an axis is clicked...
            return

        # we want to allow other navigation modes as well. Only act in case
        # shift was pressed and the correct mouse button was used
        if event.key != 'shift' or event.button != 1:
            return

        if zoomed_axes[0] is None:
            # not zoomed so far. Perform zoom

            # store the original position of the axes
            zoomed_axes[0] = (ax, ax.get_position())
            ax.set_position([0.1, 0.1, 0.85, 0.85])

            # hide all the other axes...
            for axis in event.canvas.figure.axes:
                if axis is not ax:
                    axis.set_visible(False)

        else:
            # restore the original state

            zoomed_axes[0][0].set_position(zoomed_axes[0][1])
            zoomed_axes[0] = None

            # make other axes visible again
            for axis in event.canvas.figure.axes:
                axis.set_visible(True)

        # redraw to make changes visible.
        event.canvas.draw()

    figure.canvas.mpl_connect('button_press_event', on_click)




def make_2d_score_map(data, do_smoothing, smooth_radius):
    '''
    Makes a 2d map of scores with optional smoothing
    
    Parameters:
    * ndarray (ndarray with shape Nx3)
    * do_smoothing (bool)
    * smooth_radius (int)
    
    Returns:
    * list of 2D-ndarray (for each score category one)
    '''
    
    assert data.shape[1] == 3
    
    # Make NN index
    rz_nn = NearestNeighbors(n_jobs=8)
    rz_nn.fit(data[:,0:2])
    
    # Select random elements to plot (to avoid to long computations)
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)
    idxs = idxs[0:min(len(data),100000)]
    
    # Smooth plot
    if do_smoothing:
        logging.info("Smoothing plot")
        neighbors = rz_nn.radius_neighbors(data[idxs][:,0:2],radius=smooth_radius,return_distance=False)
        
        progress_bar = ProgressBar(len(idxs))
        
        for idx, nbs in zip(idxs, neighbors):
            if len(nbs) > 0:
                data[idx,2] = sum([ data[n,2] for n in nbs ]) / len(nbs)
                
            progress_bar.print_bar()    
    
    # Throw away all not selected entries
    data = data[idxs]
    
    # Sort into categories
    categories = [0,1,2,3,5,10]
    mapdata = []
    
    for i in range(len(categories)-1):
        mask = np.greater_equal(data[:,2], categories[i]) & np.less(data[:,2], categories[i+1])
        mapdata.append(data[mask][:,0:2])
    
    mask = np.greater_equal(data[:,2], categories[-1])
    mapdata.append(data[mask][:,0:2])
    
    return mapdata





def evaluate_and_plot(tracks_edges_start, tracks_params, 
                      tracks_edges_target, history, 
                      evaluate_edge_fn, 
                      smooth_rzmap=True, smooth_radius=35.0, figsize=(16,10)):
    '''
    Function that makes a collection of plots for evaluation of a model
    
    Parameters:
    * tracks_edges: a list of ndarrays with shape [track_length,2]
    * tracks_params: a list of ndarrays with shape [track_length,selected_params]
    * history: a history dictionary (must contain the key 'loss' and 'val_loss')
    * evaluate_edge: a callable to invoke to evaluate a specific edge
    * [OPT] figsize: a tuple (width,height) in inches (?) for the figure
    
    Returns:
    * matploglib figure
    * axes array (containing 3x2 subplots)
    * score (ratio of overall 'in1' predictions)
    '''
    assert len(tracks_edges_start) == len(tracks_params) == len(tracks_edges_target)
    assert tracks_edges_start[0].shape == tracks_edges_target[0].shape
    #assert tracks_params[0].shape[1] == 3 or tracks_params[0].shape[1] == 4
    
    ##############
    # Evaluation #
    ##############
    
    logging.info("Started evaluation of test data")
    
    max_track_length = max([ len(track) for track in tracks_edges_start ])
    total_num_edges = sum([ len(track) for track in tracks_edges_start ])
    
    # Initialize the result
    EvaluationResult = collections.namedtuple("EvaluationResult", ["score_matrix", "beampipe_scores", "score_spacepoint_list"])
    
    columns = get_score_categories() + ['num_edges','num_edges_max','num_edges_min','relative_score']
    result = EvaluationResult(
        pd.DataFrame(data = np.zeros((max_track_length, len(columns))),
                     index=np.arange(max_track_length),
                     columns=columns),
        { category: [] for category in get_score_categories() },
        []
    )
    
    # Loop over all tracks
    times = []
        
    progress_bar = ProgressBar(len(tracks_edges_start))
    
    for i, (starts, targets, params) in enumerate(zip(tracks_edges_start, tracks_edges_target, tracks_params)):  
                
        for pos_in_track, (start, target, param) in enumerate(zip(starts, targets, params)):
            result = evaluate_edge_fn(pos_in_track, start, target, param, result)
            
        progress_bar.print_bar()
    
    # Normalize
    hits_per_pos = np.sum(result.score_matrix[get_score_categories()].values, axis=1)
    total_num_edges = np.sum(result.score_matrix[get_score_categories()].values.flatten())
    
    assert hits_per_pos.all() > 0
    assert total_num_edges > 0
    
    result.score_matrix['relative_score'] = result.score_matrix['relative_score'].to_numpy() / hits_per_pos
    result.score_matrix['num_edges'] = result.score_matrix['num_edges'].to_numpy() / hits_per_pos
    
    score_spacepoint_df = pd.DataFrame(columns=['pos_in_track', 'x', 'y', 'z', 'score'], data=result.score_spacepoint_list)
    
    ############
    # Plotting #
    ############
    
    logging.info("Process data and plot them")
    
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
    for i, (name, series) in enumerate(result.score_matrix[get_score_categories()].iteritems()):
        res = np.sum(series.to_numpy())
        norm_res = res / total_num_edges
        rects = ax[0,2].bar(i,norm_res,color=get_colors()[i])
        autolabel(ax[0,2], rects)
        
    ax[0,2].set_title("Score overview")
    ax[0,2].set_ylabel("fraction of test samples")
    ax[0,2].xaxis.set_visible(False)
    ax[0,2].set_ylim([0,1])
    ax[0,2].legend(get_score_categories())
    
    # Plot distribution per track pos
    current = np.zeros(max_track_length)
    for i, (name, series) in enumerate(result.score_matrix[get_score_categories()].iteritems()):
        # dont plot 'other'
        if i == len(get_score_categories()) - 1 :
            break
        
        current += series.to_numpy() / hits_per_pos
        ax[1,0].plot(current, color=get_colors()[i])
        
    ax[1,0].set_title("Score (per track position)")
    ax[1,0].set_xlabel("track position")
    ax[1,0].set_ylabel("score")
    ax[1,0].set_ylim([0,1.1])
    ax[1,0].legend(get_score_categories())
        
    # Plot beampipe score historgram
    surf_coords = np.unique(np.hstack(list(result.beampipe_scores.values())))
    surf_dict = { zcoord: i for i, zcoord in enumerate(surf_coords) }
    for k in result.beampipe_scores.keys():
        result.beampipe_scores[k] = [ surf_dict[z] for z in result.beampipe_scores[k] ]
    
    hist_data = np.array([ result.beampipe_scores[k] for k in result.beampipe_scores.keys() ], dtype=object)
    ax[1,1].set_title("Score at beampipe (wrt z coord)")
    ax[1,1].set_xlabel("z-coord bins")
    ax[1,1].set_ylabel("score")
    ax[1,1].hist(hist_data, len(surf_coords), histtype='bar', stacked=True, color=get_colors())
    ax[1,1].legend(result.beampipe_scores.keys())
    
    # Plot possible edges per track pos (DEACTIVATED IN FAVOR OF R-Z-MAP
    #ax[1,2].set_title("Mean graph edges per track position")
    #ax[1,2].set_xlabel("track position")
    #ax[1,2].set_ylabel("edges")
    #if not result.score_matrix['num_edges'].to_numpy().all() == 0:
        #ax[1,2].plot(result.score_matrix['num_edges'].to_numpy())
        #ax[1,2].plot(result.score_matrix['num_edges_max'].to_numpy())
        #ax[1,2].plot(result.score_matrix['num_edges_min'].to_numpy())
        #ax[1,2].legend(['mean', 'max', 'min'])
    #else:
        #ax[1,2].text(0.2,0.5,"no relative_score to plot")
        
        
    # R-Z-Map
    
    
    # Plot
    xy_coords = score_spacepoint_df[['x','y']].to_numpy().astype(float)
    r_coords = np.sqrt(xy_coords[:,0]**2 + xy_coords[:,1]**2)
    rzmap_data = np.vstack([
        r_coords,
        score_spacepoint_df['z'].to_numpy().astype(float),
        score_spacepoint_df['score'].to_numpy().astype(float),
    ]).T
    
    
    rzmap = make_2d_score_map(rzmap_data, smooth_rzmap, smooth_radius)    
    ax[1,2].set_title("Smoothed r-z plot" if smooth_rzmap else "r-z plot")
    ax[1,2].set_xlabel("z")
    ax[1,2].set_ylabel("r")
    for i in range(len(rzmap)):
        ax[1,2].scatter(rzmap[i][:,1], rzmap[i][:,0], color=get_colors()[i], s=5)
    
    add_subplot_zoom(fig)
    
    return fig, ax, float(np.sum(result.score_matrix[get_score_categories()[0]].to_numpy())/total_num_edges), score_spacepoint_df




def fill_in_results(pos_in_track, score, surface_z_coord, trk_params, result, num_targets=None):
    '''
    Fillst the entries of the EvaluationResult named tuple.
    
    Parameters:
    * pos_in_track: integer
    * score: in which of ['in1','in2','in3','in5','in10','other'] to put the result
    * surface_z_coord: z coord of the surface_z_coord
    * trk_params (ndarray): track parameters
    * result: EvaluationResult
    * [OPT] num_targets: integer
    
    Returns:
    * EvaluationResult: modified result type
    '''
    
    cs = get_score_categories()
    
    if score == 0: 
        result.score_matrix.loc[pos_in_track, cs[0]] += 1
        if pos_in_track == 0: result.beampipe_scores[cs[0]].append( surface_z_coord )
    elif score == 1: 
        result.score_matrix.loc[pos_in_track, cs[1]] += 1
        if pos_in_track == 0: result.beampipe_scores[cs[1]].append( surface_z_coord )
    elif score == 2: 
        result.score_matrix.loc[pos_in_track, cs[2]] += 1
        if pos_in_track == 0: result.beampipe_scores[cs[2]].append( surface_z_coord )
    elif score < 5: 
        result.score_matrix.loc[pos_in_track, cs[3]] += 1
        if pos_in_track == 0: result.beampipe_scores[cs[3]].append( surface_z_coord )
    elif score < 10:
        result.score_matrix.loc[pos_in_track, cs[4]] += 1
        if pos_in_track == 0: result.beampipe_scores[cs[4]].append( surface_z_coord )
    else: 
        result.score_matrix.loc[pos_in_track, cs[5]] += 1
        if pos_in_track == 0: result.beampipe_scores[cs[4]].append( surface_z_coord )
    
    if num_targets != None:
        result.score_matrix.loc[pos_in_track, 'num_edges'] += num_targets
        result.score_matrix.loc[pos_in_track, 'relative_score'] += 1 - score/num_targets
        
        result.score_matrix.loc[pos_in_track, 'num_edges_max'] = \
            max(result.score_matrix.loc[pos_in_track, 'num_edges_max'], num_targets)
        result.score_matrix.loc[pos_in_track, 'num_edges_min'] = \
            min(result.score_matrix.loc[pos_in_track, 'num_edges_min'], num_targets)
        
    #idx = len(result.rzmap.index)
    #result.rzmap.loc[idx] = [ pos_in_track, np.sqrt(trk_params[0]**2 + trk_params[1]**2), trk_params[2], score ]
    result.score_spacepoint_list.append((pos_in_track, trk_params[0], trk_params[1], trk_params[2], score))
    
    return result
