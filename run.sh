#!/bin/bash

module load python/3.8.6

cd /home/benjamin/Dokumente/acts_project/ml_navigator/models

python pairwise_score_navigator_pre.py --epochs 300 --prop_data_size 512 --show 0 --embedding_dim 10 --bpsplit_method uniform

python pairwise_score_navigator_pre.py --epochs 300 --prop_data_size 512 --show 0 --embedding_dim 10 --bpsplit_method density

python pairwise_score_navigator_pre.py --epochs 300 --prop_data_size 512 --show 0 --embedding_dim 10 --use_real_space_as_embedding 1
