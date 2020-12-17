import random
import datetime
import os
import logging
import pprint

import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from feedforward_navigator import build_feedforward_model
from common.preprocessing import *
from common.plotting import *


#############################
# The core of the algorithm #
#############################

def build_and_fit_feedforward(params, x_train, y_train, validation_split = 0.3):
    '''
    takes params and training data, returns (history, model)
    '''
    
    model_params = {
        'embedding_dim': params['embedding_dim'],
        'hidden_layers': [ params['layer_size'] ] * params['network_depth'],
        'learning_rate': params['learning_rate']
    }
    
    model = build_feedforward_model(**model_params)
    
    assert len(x_train[0]) == len(x_train[1]) == len(y_train)
    
    # Don't need track substructure for feedforward
    x_train_emb = np.concatenate(x_train[0])
    x_train_par = np.concatenate(x_train[1])
    y_train = np.concatenate(y_train)
    
    
    # shuffle TODO somehow do this outside for performance reasons
    idxs = np.arange(len(x_train[0]))
    np.random.shuffle(idxs)
    
    x_train_emb = x_train_emb[idxs]
    x_train_par = x_train_par[idxs]
    y_train = y_train[idxs]
    
    history = model.fit(
        x=[ x_train_emb, x_train_par ], 
        y=y_train,
        validation_split=validation_split,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=0
    )

    return history, model


##########################
# Evolutionary functions #
##########################

def evaluate_networks(population, x_train, y_train, show_progress_bar):
    val_losses = []
    models = []
        
    if show_progress_bar:
        print_progress_bar(0, len(population))
    
    for i, params in enumerate(population):  
        if show_progress_bar:
            print_progress_bar(i+1, len(population))
            
        tf.keras.backend.clear_session()
        history, model = build_and_fit_feedforward(params, x_train, y_train)
        
        models.append(model)
        val_losses.append(history.history['val_loss'][-1])
    
    logging.info("finished build and fit")
        
    # sort by validation losses
    val_losses = np.array(val_losses)
    idxs = np.argsort(val_losses)
    population = np.array(population)[idxs].tolist()
    
    logging.info("generation statistics: loss=%f+-%f (best=%f, worst=%f)", \
                 np.mean(val_losses), np.std(val_losses), val_losses[idxs[0]], val_losses[idxs[-1]])
    
    return population, models[idxs[0]], population[idxs[0]]


def select_surviving(population, best_select, random_select):
    '''
    Filters out most of the old population. Assumes the population is sorted by score
    '''
    assert best_select >= 0.0 and best_select <= 1.0
    assert random_select >= 0.0 and random_select <= 1.0
    
    num_best = int(best_select*len(population))
    
    new_population = population[0:num_best]
    new_population += [ m for m in population[num_best:] if random.random() < random_select ]
    
    logging.info("surviving this generation: the %d best + %d of the rest",num_best,len(new_population)-num_best)
    
    return new_population


def mutate(population, mutate_chance, keys, parameter_ranges):
    for member in population:
        if random.random() < mutate_chance:
            key = random.choice(keys)
            member[key] = random.choice(parameter_ranges[key])
        
    return population


def breed(population, population_size, keys):
    parents = population
    children = []
    
    while len(children) < population_size - len(parents):
        mother = random.choice(parents)
        father = random.choice(parents)
        
        child = mother
        
        for key in keys:
            child[key] = random.choice([ mother[key], father[key] ])
            
        children.append(child)
        
    return parents + children


#################
# Main function #
#################

def main():
    ##############################################
    # Setup logging and printing, check hardware #
    ##############################################

    tf.get_logger().setLevel('ERROR')
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
        )
        
    logging.info("tensorflow devices: %s",[ t for n, t in tf.config.list_physical_devices() ])
    
    ##########################
    # Generate training data #
    ##########################

    root_dir = "/home/benjamin/Dokumente/acts_project/ml_navigator/data/"
    propagation_file = root_dir + "logger/navigation_training/data-201113-151732-n1.csv"
    embedding_file = root_dir + "models/embeddings/20201201-142508-emb10-acc36"
    detector_file = root_dir + "detector/detector_surfaces.csv"

    x_train_embs, x_train_pars, y_train_embs, nn = prepare_data(propagation_file, embedding_file, detector_file)

    embedding_dim = x_train_embs[0].shape[1]
    logging.info("embedding dimension: %d",embedding_dim)

    x_train = [x_train_embs, x_train_pars]
    y_train = y_train_embs

    #########################
    # Initialize population #
    #########################

    parameter_ranges = {
        'network_depth': np.arange(1,10).tolist(),
        'layer_size': np.arange(50,1500,10).tolist(),
        #'network_depth': np.arange(1,3).tolist(),
        #'layer_size': np.arange(50,250,10).tolist(),
        'learning_rate': np.logspace(-4,0,10).tolist(), # 10^-4 -> 10^0
        'batch_size': [ 128, 256, 512, 1024, 2048, 4096, 8192 ]
    }

    keys = list(parameter_ranges.keys())


    population_size = 40
    population = []

    for i in range(population_size):
        params = {}
        
        for key in keys:
            params[key] = random.choice(parameter_ranges[key])
        
        params['epochs'] = 200
        #params['epochs'] = 5
        params['embedding_dim'] = embedding_dim
        
        population.append(params)

    ##############################
    # Run evolutionary algorithm #
    ##############################

    best_select = 0.4
    random_select = 0.1
    mutate_chance = 0.2

    generations = 10

    best_model = None
    best_params = None

    for gen in range(generations):
        logging.info("START GENERATION %d",gen)
        
        population, model, params = evaluate_networks(population, x_train, y_train, show_progress_bar=True)
        population = select_surviving(population, best_select, random_select)
        population = mutate(population, mutate_chance, keys, parameter_ranges)
        population = breed(population, population_size, keys)
        
        best_model = model
        best_params = params
        assert len(population) == population_size

    logging.info("best parameters:")
    pprint.pprint(best_params)

    date_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = date_string + "-forward"
    model.save(root_dir + "models/navigator/evolutionary/" + output_filename)
    
    
    
if __name__ == "__main__":
    main()
