import os

import numpy as np
#np.set_printoptions(precision=3)
#np.set_printoptions(suppress=True)

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split



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



def make_embedding_network(num_categories,embedding_dim,hidden_layers):
    input_node_1 = tf.keras.Input(1)
    input_node_2 = tf.keras.Input(1)
    
    embedding_layer = tf.keras.layers.Embedding(num_categories,embedding_dim)
    
    x = embedding_layer(input_node_1)
    y = embedding_layer(input_node_2)
    
    z = tf.keras.layers.Concatenate()([x,y])
        
    z = tf.keras.layers.Dense(hidden_layers[0])(z)
    z = tf.keras.layers.Activation(tf.nn.relu)(z)
    
    for i in range(1,len(hidden_layers)):
        z = tf.keras.layers.Dense(hidden_layers[i])(z)
        z = tf.keras.layers.Activation(tf.nn.relu)(z)
        
    z = tf.keras.layers.Dense(1)(z)
    output_node = tf.keras.layers.Activation(tf.nn.sigmoid)(z)
    
    return tf.keras.Model(inputs=[input_node_1,input_node_2],outputs=[output_node])

######################
# PREPARE GRAPH DATA #
######################

graph_data = pd.read_csv("build/data.csv", dtype={'start_id': np.uint64, 'end_id': np.uint64})

geoid_edges = graph_data[["start_id","end_id"]].to_numpy()
positions = graph_data[["pos_x","pos_y","pos_z"]].to_numpy()

# Nodes
nodes = np.unique(geoid_edges.flatten())
print("num total nodes:",len(nodes))

geoid_dict = { geo_id : number for geo_id,number in zip(nodes, np.arange(len(nodes))) }

# Edges (423485)
geoid_edges, weights = np.unique(geoid_edges,axis=0,return_counts=True)
print("num total edges:",len(geoid_edges))
weights = weights.astype(np.float32)

# Transform geoids to numbers starting from 0
ordinal_edges = np.empty(shape=geoid_edges.shape,dtype=np.int32)
for o_edge, id_edge in zip(ordinal_edges,geoid_edges):
    o_edge[0] = geoid_dict[id_edge[0]]
    o_edge[1] = geoid_dict[id_edge[1]]

# Use log to 'smooth' the weights, then norm to 1, log should enable reweighting without suppressing most to zero
weights = np.log(weights+1)
weights /= np.max(weights)

##########################
# GENERATE TRAINING DATA #
##########################

num_samples = 250000
assert num_samples > len(ordinal_edges)

print("true edges ration:",100*len(ordinal_edges)/num_samples,"%")

unconnected_edges = np.zeros(shape=(num_samples-len(ordinal_edges),2),dtype=np.int32)

# use set for fast lookup if an edge already exists
edge_set = set([(e[0],e[1]) for e in ordinal_edges])

# generate random edges that are not connected (y = 0)
for i in range(len(unconnected_edges)):
    while True:
        new_sample = np.random.randint(0,len(nodes),2).astype(np.int32)
        
        if new_sample[0] != new_sample[1] and (new_sample[0],new_sample[1]) not in edge_set:
            unconnected_edges[i] = new_sample
            edge_set.add((new_sample[0],new_sample[1]))
            break

samples_x = np.concatenate([ordinal_edges,unconnected_edges])
samples_y = np.concatenate([weights,np.zeros(len(unconnected_edges))])

assert len(samples_x) == len(samples_y)
assert len(samples_x) == len(np.unique(samples_x,axis=0))


###################
# LEARN EMBEDDING #
###################

print("Start learning embedding",flush=True)

embedding_model_params = {
    "num_categories": len(nodes),
    "embedding_dim": 10,
    "hidden_layers": [50,50]
}

embedding_model = make_embedding_network(**embedding_model_params)

embedding_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
    loss=tf.keras.losses.MeanSquaredError()
)

embedding_model.summary()
    
embedding_model.fit(
    x=[ samples_x[:,0], samples_x[:,1] ], 
    y=samples_y, 
    batch_size=32,
    epochs=50, 
    validation_split=0.33,
    verbose=2
)
    
