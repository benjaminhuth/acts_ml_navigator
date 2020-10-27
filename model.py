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
    if output_activation:
        output_node = tf.keras.layers.Activation(output_activation)(x)
        
    return tf.keras.Model(input_node, output_node)


class NavigationModel:
    def __init__(self,num_categories,embedding_dim,dense_layers):
        # embeddes the ids in a higher-dimensional space
        self.embedding_network = make_embedding_network(num_categories,embedding_dim)
        
        # takes direction, position and two surface embeddings: result [0,1] How likely these surfaces are connected?
        self.network = make_keras_mlp(3 + 3 + 2*embedding_dim, dense_layers, 1, output_activation=tf.nn.sigmoid)
        
    def forward(self,position,direction,start_surface_id,end_surface_id):
        # create embeddings
        embeddings = self.embedding_network([start_surface_id,end_surface_id])
        
        # combine input data
        input_data = tf.concat([position,direction,embeddings[0],embeddings[1]],0)
        
        return self.network(input_data)
    
    def train(self,train_x, train_y):
        pass


######################
# PREPARE GRAPH DATA #
######################

graph_data = pd.read_csv("build/data.csv")

edges = graph_data[["start_id","end_id"]].to_numpy()
positions = graph_data[["pos_x","pos_y","pos_z"]].to_numpy()

# Edges
print("num total edges:",len(edges))
edges, weights = np.unique(edges,axis=0,return_counts=True)
