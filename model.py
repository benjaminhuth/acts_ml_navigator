import os

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def make_embedding_network(num_categories,embedding_dim):
    input_node = tf.keras.Input(1)
    x = tf.keras.layers.Embedding(num_categories,embedding_dim)(input_node)
    output_node = tf.keras.layers.Flatten()(x)
    
    return tf.keras.Model(input_node, output_node)

def make_keras_mlp(input_shape, hidden_layers, output_shape, output_activation=None):
    assert len(hidden_layers) >= 1
    
    input_node = tf.keras.Input(input_shape)
    
    x = tf.keras.layers.Dense(hidden_layers[0])(input_node)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    
    for i in range(1,len(hidden_layers)):
        x = tf.keras.layers.Dense(hidden_layers[i])(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        
    x = tf.keras.layers.Dense(output_shape)(x)
    
    output_node = x
    if output_softmax:
        output_node = tf.keras.layers.Activation(tf.nn.softmax)(x)
        
    return tf.keras.Model(input_node, output_node)


class NavigationModel:
    def __init__(num_categories,embedding_dim,dense_layers):
        # embeddes the ids in a higher-dimensional space
        self.embedding_network = make_embedding_network(num_categories,embedding_dim)
        
        # takes direction, position and two surface embeddings: result [0,1] How likely these surfaces are connected?
        self.network = make_keras_mlp(3 + 3 + 2*embedding_dim, dense_layers, 1, output_activation=tf.nn.sigmoid)
        
    def forward(position,direction,start_surface_id,end_surface_id):
        # create embeddings
        embeddings = self.embedding_network([start_surface_id,end_surface_id])
        
        # combine input data
        input_data = tf.concat([position,direction,embeddings[0],end_embedding[1]],0)
        
        return self.network(input_data)
    
    def train(train_x, train_y):
        pass
     

#########################
# PREPARE DETECTOR DATA #
#########################

detector_data = pd.read_csv("generic_detector.csv")
geo_ids = detector_data["geometry_id"].to_numpy()
surface_positions = detector_data[["cx","cy","cz"]].to_numpy()

# assign each geo id a number (because geo_ids go not from 0 to N)
numbers = np.arange(len(geo_ids))
geo_dict = { key : value for value, key in zip(numbers,geo_ids) }

# position dictionary
pos_dict = { geo_id : pos for geo_id, pos in zip(geo_ids,surface_positions) }


######################
# PREPARE GRAPH DATA #
######################

graph_data = pd.read_csv("build/data.csv")

edges = graph_data[["start_id","end_id"]].to_numpy()

print("num total edges:",len(edges))

edges, weights = np.unique(edges,axis=0,return_counts=True)

print("num unique edges:",len(edges))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for edge in edges:
    try:
        a = pos_dict[edge[0]]
        b = pos_dict[edge[1]]
        
        ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]])
    except:
        continue
    
plt.show()
