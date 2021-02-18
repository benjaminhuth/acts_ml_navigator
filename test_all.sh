#!/bin/bash

module load python/3.8.6

TEST_DIR=/tmp/mlnav_testdir
MINIMAL_SETTINGS="--disable_gpu=1 --prop_data_size=1 --epochs=1 --export=1 --show=0 --output_dir=$TEST_DIR"

#######################
# Make test directory #
#######################

mkdir -p $TEST_DIR
(cd $TEST_DIR && rm -rf *)

#################
# Run all tests #
#################

SCRIPT=$PWD/models/pairwise_score_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --data_gen_method=graph

if [ $? -ne 0 ]; then
    printf "\n>>>>>> ERROR OCCURED <<<<<<\n\n"
    rm -rf $TEST_DIR
    exit 1
fi

(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"

#########################################################



SCRIPT=$PWD/models/pairwise_score_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --data_gen_method=false_sim

if [ $? -ne 0 ]; then
    printf "\n>>>>>> ERROR OCCURED <<<<<<\n\n"
    rm -rf $TEST_DIR
    exit 1
fi

(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"

#########################################################



SCRIPT=$PWD/models/target_pred_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --evaluation_method=graph

if [ $? -ne 0 ]; then
    printf "\n>>>>>> ERROR OCCURED <<<<<<\n\n"
    rm -rf $TEST_DIR
    exit 1
fi

(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"

#########################################################



SCRIPT=$PWD/models/target_pred_navigator_pre.py
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS --evaluation_method=nn

if [ $? -ne 0 ]; then
    printf "\n>>>>>> ERROR OCCURED <<<<<<\n\n"
    rm -rf $TEST_DIR
    exit 1
fi

(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"

#########################################################


SCRIPT=$PWD/models/weighted_graph_navigator.py --bpsplit_z 20
printf "=====| TEST $SCRIPT |=====\n\n"
python $SCRIPT $MINIMAL_SETTINGS

if [ $? -ne 0 ]; then
    printf "\n>>>>>> ERROR OCCURED <<<<<<\n\n"
    rm -rf $TEST_DIR
    exit 1
fi

(cd $TEST_DIR && ls -l && rm -rf *)
printf "\n"


#########################
# Remove test directory #
#########################

rm -rf $TEST_DIR
