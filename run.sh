#!/bin/bash

module load python/3.8.6

cd /home/benjamin/Dokumente/acts_project/ml_navigator/models

python pairwise_score_navigator_self \
    --learning_rate=0.001 --epochs 100 --prop_data_size 128

# python graph_forward_navigator.py --network_depth=3 --layer_size=500 --learning_rate=0.001
