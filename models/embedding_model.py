import os
import sys
import datetime

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt
from utility.data_import import *

######################
# PREPARE GRAPH DATA #
######################

def main(argv):
    nodes, geoid_edges, weights = generate_graph_from_data("../data/logger/data-201104-120629.csv")
    
    print("num nodes:",len(nodes))
    print("num edges:",len(geoid_edges))
    
    # get all valid nodes
    all_nodes = get_list_of_all_geoids("../data/event000000000-detectors.csv")
    print("detector coverage ratio:",len(nodes)/len(all_nodes))
    print("surfaces missing:",len(all_nodes)-len(nodes))

    # Transform geoids to numbers starting from 0
    geoid_dict = { geo_id : number for geo_id,number in zip(all_nodes, np.arange(len(all_nodes))) }

    connected_edges = np.empty(shape=geoid_edges.shape,dtype=np.int32)
    for o_edge, id_edge in zip(connected_edges,geoid_edges):
        o_edge[0] = geoid_dict[id_edge[0]]
        o_edge[1] = geoid_dict[id_edge[1]]

    ##########################
    # GENERATE TRAINING DATA #
    ##########################
    
    num_samples = 350000
    assert num_samples > len(connected_edges)
    print("true edges ration:",100*len(connected_edges)/num_samples,"%")

    # generate random edges that are not connected (y = 0)
    unconnected_edges = np.zeros(shape=(num_samples-len(connected_edges),2),dtype=np.int32)
    edge_set = set([(e[0],e[1]) for e in connected_edges])
    
    for i in range(len(unconnected_edges)):
        while True:
            new_sample = np.random.randint(0,len(all_nodes),2).astype(np.int32)
            
            if new_sample[0] != new_sample[1] and (new_sample[0],new_sample[1]) not in edge_set:
                unconnected_edges[i] = new_sample
                edge_set.add((new_sample[0],new_sample[1]))
                break

    # combine data
    samples_x = np.concatenate([connected_edges,unconnected_edges])    
    samples_y = np.concatenate([np.ones(len(connected_edges)),np.zeros(len(unconnected_edges))])
    sample_weights = np.concatenate([weights,np.ones(len(unconnected_edges))])

    assert len(samples_x) == len(sample_weights)
    assert len(samples_x) == len(samples_y)
    assert len(samples_x) == len(np.unique(samples_x,axis=0))

    print("samples_x.shape:",samples_x.shape)
    print("samples_y.shape:",samples_y.shape)

    ###################
    # LEARN EMBEDDING #
    ###################

    print("Start learning embedding",flush=True)
    
    def build_model(num_categories,embedding_dim,hidden_layers,learning_rate):
        input_node_1 = tf.keras.Input(1)
        input_node_2 = tf.keras.Input(1)
        
        embedding_layer = tf.keras.layers.Embedding(num_categories,embedding_dim)
        
        x = embedding_layer(input_node_1)
        y = embedding_layer(input_node_2)
        
        z = tf.keras.layers.Concatenate()([x,y])
        
        for i in range(len(hidden_layers)):
            z = tf.keras.layers.Dense(hidden_layers[i])(z)
            z = tf.keras.layers.Activation(tf.nn.relu)(z)
            
        z = tf.keras.layers.Dense(1)(z)
        output_node = tf.keras.layers.Activation(tf.nn.sigmoid)(z)
        
        model = tf.keras.Model(inputs=[input_node_1,input_node_2],outputs=[output_node])
        
        model.summary()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Accuracy()]
        )
        
        return model

    model_params = {
        'num_categories': len(all_nodes),
        'embedding_dim': 10,
        'hidden_layers': [50],
        'learning_rate': 0.01
    }
    
    train_embedding_model = build_model(**model_params)

    history = train_embedding_model.fit(
        x=[ samples_x[:,0], samples_x[:,1] ], 
        y=samples_y, 
        #sample_weight=sample_weights,
        batch_size=256,
        epochs=300,
        verbose=2
    )
    
    do_plot = False
    if do_plot:
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()    
    
    # Embedding model
    input_node = tf.keras.Input(1)
    output_node = train_embedding_model.get_layer("embedding")(input_node)
    
    embedding_model = tf.keras.Model(input_node,output_node)
    
    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_string = "-emb" + str(model_params['embedding_dim'])
    accuracy_string = "-acc" + str(round(history.history['accuracy'][-1]*100))
    output_filename = "../data/embeddings/" + date_string + embedding_string + accuracy_string
        
    embedding_model.save(output_filename)
    print("exported model to '",output_filename,"'")
    
    
if __name__ == "__main__":
    main(sys.argv)
