#!/bin/bash

module load python/3.8.6

cd /home/benjamin/Dokumente/acts_project/ml_navigator/models

python nn_forward_navigator.py \
    --learning_rate=0.001 --layer_size=700 --network_depth=4 --validation_split=0.1 --test_split=0.1 --epochs 100 --batch_size 8196

# python graph_forward_navigator.py --network_depth=3 --layer_size=500 --learning_rate=0.001
