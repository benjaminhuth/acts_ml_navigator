import os
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from common.plotting import *

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


class NeighborDistributionCallback(tf.keras.callbacks.Callback):
    '''
    Callback class, which invokes neighbor_accuracy_detailed(...) after each epoch
    '''
    def __init__(self, x, y, neighbor_index, sample_rate=10, do_print=True):
        self.nn = neighbor_index
        self.x = x
        self.y = y
        self.sample_rate = sample_rate
        self.do_print = do_print
    
    def compute_accuray(self):
        y_pred = self.model.predict(self.x)
        return neighbor_accuracy_detailed(self.y, y_pred, self.nn)
    
    def on_train_begin(self, logs=None):
        self.results = []
        self.log_epochs = []
        
    def print_accuracy(self, acc):
        print("neigbor acc - in1: {:.2f} - in2: {:.2f} - in3 {:.2f} - in5 {:.2f} - in10 {:.2f} - other {:.1f}".format(*acc), flush=True)
        
    def on_epoch_end(self, epoch, logs=None): 
        if epoch % self.sample_rate == 0:
            acc = self.compute_accuray()
            self.results.append(acc)
            self.log_epochs.append(epoch)

            if self.do_print:
                self.print_accuracy()
        
    def on_train_end(self, logs=None):
        acc = self.compute_accuray()
        self.results.append(acc)
        self.log_epochs.append(self.params['epochs']-1)
            
        if self.do_print:
            self.print_accuracy()
        
        self.model.history.history['neighbor_accuracy_detailed'] = (self.log_epochs, np.asarray(self.results))
        
        

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

def make_evaluation_plot(y_true, y_pred, nn, track_lengths, history):
    '''
    Returns (fig, axes)
    '''
    assert np.sum(track_lengths) == len(y_pred)
    
    idxs_1st = np.array([ np.sum(track_lengths[0:i]) for i in range(len(track_lengths)) ])
    idxs_2nd = idxs_1st + 1
    idxs_last = np.roll(idxs_1st - 1, -1)
        
    results_1st = neighbor_accuracy_detailed(y_true[idxs_1st], y_pred[idxs_1st],nn)
    results_last = neighbor_accuracy_detailed(y_true[idxs_last], y_pred[idxs_last],nn)
    results_all = neighbor_accuracy_detailed(y_true, y_pred,nn)
    
    results = [results_all, results_1st, results_last]
    titles = ["All connections", "1st edge of track", "Last edge of track"]
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.set_figheight(9)
    fig.set_figwidth(12)
    
    colors = get_colors()
    
    for ax, result, title in zip(axes[0,:], results, titles):
        
        for i, res in enumerate(result):
            rects = ax.bar(i,res,color=colors[-(i+1)])
            autolabel(ax, rects)
        
        ax.set_title(title)
        ax.set_ylim([0.0,1.1])
        ax.set_xticks([0,1,2,3,4,5])
        ax.set_xticklabels(["1","2","3","5","10","other"])
    
    
    # find out which index corresponds to what "position in track"
    pos_list = []
    pos_in_track = 0
    n_track = 0
    
    for i in range(len(y_true)):
        if pos_in_track == track_lengths[n_track]:
            pos_in_track = 0
            n_track += 1
            
        pos_list.append(pos_in_track)
        pos_in_track += 1
    
    pos_list = np.array(pos_list)
    
    # Compute position wise accuracy
    best_1 = []
    
    for i in range(np.max(track_lengths)):
        mask = np.equal(pos_list, i)
        best_1.append(next_neighbor_accuracy(y_true[mask], y_pred[mask], nn)) 
    
    
    axes[1,0].set_title("Accuracy per track-position")
    axes[1,0].plot(best_1,color=colors[-1],linewidth=2.0)
    axes[1,0].set_ylim([0.0,1.1])
    
    # Plot training history
    axes[1,1].set_title("Training history")
    axes[1,1].plot(history.history["loss"])
    axes[1,1].plot(history.history["val_loss"])
    #axes[1,1].set_yscale('log')
    axes[1,1].legend(["train loss", "validation loss"])
    #axes[1,1].minorticks_off()
    #axes[1,1].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    #axes[1,1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    # Turn off last square
    axes[1, 2].axis('off')
    
    return fig, axes
