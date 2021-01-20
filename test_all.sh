#!/bin/bash

module load python/3.8.6

TEST_DIR=/tmp/mlnav_testdir
MINIMAL_SETTINGS="--disable_gpu=1 --prop_data_size=1 --epochs=1 --export=1 --show=0 --output_dir=$TEST_DIR"


# Make test directory
mkdir -p $TEST_DIR
(cd $TEST_DIR && rm -rf *)


# Run all tests
SCRIPT=./models/pairwise_score_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --data_gen_method=graph
(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


SCRIPT=./models/pairwise_score_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --data_gen_method=false_sim
(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


SCRIPT=./models/pairwise_score_navigator_self.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS
(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


SCRIPT=./models/pairwise_score_navigator_self.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --evaluation_method=graph
(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


SCRIPT=./models/target_pred_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --evaluation_method=graph
(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


SCRIPT=./models/weighted_graph_navigator.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS
(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


# Remove test directory
rm -rf $TEST_DIR
