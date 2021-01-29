#!/bin/bash

./build/navigator/navigation_test \
    -n 100 \
    -j 1 \
    --prop-ntests 1 \
    --rnd-seed ${RANDOM} \
    --nav_model ./data/models/pairwise_score_navigator_pre/20210128-151344-emb3-graph-n512-acc77.onnx \
    --graph_data ./data/logger/navigation_training/data-201218-135649-n512.csv \
    --bpsplit_z ./data/models/embeddings/20210125-124832-emb10-acc40-bz400den-bp16.txt
