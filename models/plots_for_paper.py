import os
import logging

from tkinter import Tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

rcParams['font.serif'] = 'Computer Modern Roman'
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True
rcParams['figure.constrained_layout.use'] = True
rcParams['font.size'] = 11

from common.evaluation import make_2d_score_map, get_score_categories

Tk().withdraw()

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



logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

def score_distribution(scores):
    dist = []
    
    dist.append( np.sum( np.equal(scores, 0) ) )
    dist.append( np.sum( np.equal(scores, 1) ) )
    dist.append( np.sum( np.equal(scores, 2) ) )
    dist.append( np.sum( np.greater_equal(scores, 3) & np.less(scores, 5) ) )
    dist.append( np.sum( np.greater_equal(scores, 5) & np.less(scores, 10) ) )
    dist.append( np.sum( np.greater_equal(scores, 10) ) )
    
    return dist / sum(dist)
    
# Paths
data_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/models/"
target_pred_file = os.path.join(data_dir, "target_pred_navigator_pre/generic/20210223-171635-emb10-acc68-nn.csv")
pairwise_score_file = os.path.join(data_dir, "pairwise_score_navigator_pre/generic/20210224-123427-emb3-graph-n512-acc77.csv")
weighted_graph_file = os.path.join(data_dir, "weighted_graph_navigator/generic/20210224-114202-n512-acc33.csv")
output_dir = "/home/benjamin/Dokumente/acts_project/vChep2021/figs/"

assert os.path.exists(target_pred_file)
assert os.path.exists(pairwise_score_file)
assert os.path.exists(weighted_graph_file)
assert os.path.exists(output_dir)

# Load CSVs
target_pred_df = pd.read_csv(target_pred_file)
pairwise_score_df = pd.read_csv(pairwise_score_file)
weighted_graph_df = pd.read_csv(weighted_graph_file)


# What to do
plot_score_dist = True
plot_rzmap = True
plot_xymap = True

#############
# Bar plots #
#############

if plot_score_dist:
    # Score distribution
    target_pred_dist = score_distribution(target_pred_df['score'].to_numpy())
    pairwise_score_dist = score_distribution(pairwise_score_df['score'].to_numpy())
    weighted_graph_dist = score_distribution(weighted_graph_df['score'].to_numpy())
    
    # Plot
    width = 0.3
    x = np.arange(len(get_score_categories()))

    fig, ax = plt.subplots(figsize=(9,3))
        
    rects1 = ax.bar(x - width, target_pred_dist, width, color=get_colors(), label='target prediction', edgecolor = "black", hatch='//')
    rects2 = ax.bar(x, pairwise_score_dist, width, color=get_colors(), label='score prediction', edgecolor = "black", hatch='.')
    rects2 = ax.bar(x + width, weighted_graph_dist, width, color=get_colors(), label='weighted graph', edgecolor = "black", hatch='\\\\')
    #autolabel(ax, rects1)
    #autolabel(ax, rects2)

    ax.set_title("Score overview")
    ax.set_ylabel("fraction of test samples", fontsize=12)
    ax.set_ylim([0,1])
    ax.set_xticks(x)
    ax.set_xticklabels(get_score_categories())
    ax.legend()
    #ax.legend(prop={'size': 12})

    plt.show()
    
    #if messagebox.askokcancel("", "Save score distribution?"):
    fig.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=300)


#############
# R-Z Plots #
#############

smooth_radius = 5.
marker_size = 1.5

def make_rz_plot(df, title):
    #matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(4.5,3.5))

    xy_coords = df[['x','y']].to_numpy().astype(float)
    r_coords = np.sqrt(xy_coords[:,0]**2 + xy_coords[:,1]**2)
    rzmap_data = np.vstack([
        r_coords,
        df['z'].to_numpy().astype(float),
        df['score'].to_numpy().astype(float),
    ]).T

    rzmap = make_2d_score_map(rzmap_data, True, smooth_radius)    
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    for i in range(len(rzmap)):
        ax.scatter(rzmap[i][:,1], rzmap[i][:,0], color=get_colors()[i], s=marker_size)
        
    return fig, ax


if plot_rzmap:
    logging.info("started making rz plots")
    
    target_pred_fig, target_pred_ax = make_rz_plot(target_pred_df, r"target prediction: $r$-$z$ projection")
    logging.info("finished target pred rz plot")
    #plt.show()
    
    pairwise_score_fig, pairwise_score_ax = make_rz_plot(pairwise_score_df, r"score prediction: $r$-$z$ projection")
    logging.info("finished pairwise score rz plot")
    #plt.show()
    
    #if messagebox.askokcancel("", "Save rz plots?"):
    target_pred_fig.savefig(os.path.join(output_dir, "target_pred_rzmap.png"), dpi=300)
    pairwise_score_fig.savefig(os.path.join(output_dir, "pairwise_score_rzmap.png"), dpi=300)
    
    logging.info("saved figures")


#############
# X-Y Plots #
#############


def make_xy_plot(df, title):
    #matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(4.5,3.5))

    xymap_data = df[['x','y','score']].to_numpy().astype(float)

    xymap = make_2d_score_map(xymap_data, True, smooth_radius)    
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([-1000,-500,0,500,1000])
    ax.set_yticks([-1000,-500,0,500,1000])
    ax.set_aspect('equal')
    for i in range(len(xymap)):
        ax.scatter(xymap[i][:,1], xymap[i][:,0], color=get_colors()[i], s=marker_size)
        
    return fig, ax
    


if plot_xymap:
    logging.info("started making xy plots")
    
    target_pred_xy_fig, target_pred_xy_ax = make_xy_plot(target_pred_df, r"target prediction: $x$-$y$ projection")
    logging.info("finished target pred xy plot")
    #plt.show()
    
    pairwise_score_xy_fig, pairwise_score_xy_ax = make_xy_plot(pairwise_score_df, r"score prediction: $x$-$y$ projection")
    logging.info("finished pairwise score xy plot")
    #plt.show()
    
    #if messagebox.askokcancel("", "Save xy plots?"):
    target_pred_xy_fig.savefig(os.path.join(output_dir, "target_pred_xymap.png"), dpi=300)
    pairwise_score_xy_fig.savefig(os.path.join(output_dir, "pairwise_score_xymap.png"), dpi=300)
    
    logging.info("saved figures")
