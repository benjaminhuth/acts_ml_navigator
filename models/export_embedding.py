import logging
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from common.misc import export_embedding_file
from common.config import *



if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',level=logging.INFO)    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_model_dir",type=str)
    
    options = vars(parser.parse_args())
    assert os.path.exists(options['emb_model_dir'])
    
    # Load model
    embedding_model = tf.keras.models.load_model(options['emb_model_dir'], compile=False)
    
    # Split path from filename
    if options['emb_model_dir'][-1] == '/':
        options['emb_model_dir'] = options['emb_model_dir'][:-1]
    
    path, filename = os.path.split(options['emb_model_dir'])
    
    # Extract bpsplit
    parts = filename.split('-')
    bpsplit_z = int(parts[4][2:-3])
    bpsplit_phi = int(parts[5][2:])
    
    logging.info("Embedding model has beampipe split (%d,%d)", bpsplit_z, bpsplit_phi)
    
    # Generate output filename
    filename += "-embeddings.csv"
    output_file = os.path.join(path, filename)
    
    logging.info("Will save embedding with filename '%s'",output_file)
    
    # Get detector file
    detector_file = os.path.join(get_root_dir(), "detector/detector_surfaces.csv")
    
    # Call export function
    export_embedding_file(output_file, detector_file, embedding_model, bpsplit_phi*bpsplit_z)
        
    
