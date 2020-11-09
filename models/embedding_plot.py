import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from utility.data_import import get_sorted_model_dirs


def main():
    # Sort model subdirs by accuracy
    model_dirs = get_sorted_model_dirs('../data/embeddings/')

    # Load best model
    model = tf.keras.models.load_model('../data/embeddings/' + model_dirs[0], compile=False)
    print("loaded './embeddings/" + model_dirs[-1] + "'")
    
    # Load geoid to name encoding
    geoid_name_dict = {}
    
    with open('../build/utilities/volume_name_extractor/volume_names.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # header
        for row in reader:
            # dict[geoid] = name
            geoid_name_dict[ int(row[0]) ] = row[1]
            

    # Load node number to geoid encoding
    number_geoid_dict = {}

    with open('./embeddings/' + model_dirs[-1] + '-nodes.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # dict[number] = geoid
            number_geoid_dict[ int(row[1]) ] = int(row[0])
            
    # Get nodes of different detector parts
    beamline = []
    pixel = []
    sstrip = []
    lstrip = []
    
    for n in range(len(number_geoid_dict)):
        geoid = number_geoid_dict[n]
        
        try:
            name = geoid_name_dict[geoid]
            
            if "Beamline" in name:
                beamline.append(n)
            elif "Pixel" in name:
                pixel.append(n)
            elif "SStrip" in name:
                sstrip.append(n)
            elif "LStrip" in name:
                lstrip.append(n)
        except:
            continue
            
    num_skipped = len(number_geoid_dict) - (len(beamline) + len(pixel) + len(sstrip) + len(lstrip))
    print("num of skipped geoids:",num_skipped)
    
    # Get embedding of all nodes
    node_numbers = np.array(beamline + pixel + sstrip + lstrip)
    print("node_numbers.shape:",node_numbers.shape)
    
    node_embeddings = model(node_numbers)
    node_embeddings = np.reshape(node_embeddings,newshape=(node_embeddings.shape[0],node_embeddings.shape[2]))
    print("node_embeddings.shape:",node_embeddings.shape)

    scaler = StandardScaler()
    node_embeddings = scaler.fit_transform(node_embeddings)

    # Apply MDS
    reduced_dims = 2
    mds = MDS(reduced_dims,n_init=1,max_iter=100)

    print("start MDS",flush=True)

    embeddings_reduced = mds.fit_transform(node_embeddings)
    print("embeddings_reduced.shape:",embeddings_reduced.shape)
    
    # Reconstruct different detector parts
    start = 0
    end = len(beamline)
    beamline = embeddings_reduced[start:end]
    
    start = end
    end += len(pixel)
    pixel = embeddings_reduced[start:end]
    
    start = end
    end += len(sstrip)
    sstrip = embeddings_reduced[start:end]
    
    start = end
    end += len(lstrip)
    lstrip = embeddings_reduced[start:end]
    
    # Plot the things
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    
    plt.scatter(beamline[:,0],beamline[:,1],c='y')
    plt.scatter(pixel[:,0],pixel[:,1],c='b')
    plt.scatter(sstrip[:,0],sstrip[:,1],c='r')
    plt.scatter(lstrip[:,0],lstrip[:,1],c='g')
    
    plt.show()
    
    
    
if __name__ == "__main__":
    main()
