#!/bin/bash

printf "n,i,nodes,edges,filesize_megabyte\n"

MAX_JOBS=20

for ((i = 1 ; i <= 128 ; i*=2)); do
    printf "$i,"
    
    n=0
    
    # there occure sometimes errors with high numbers of events (max path length)
    while true; do
        let n=n+1
        OUTPUT="$(../build/logger/logger -n $i --bf-values 0 0 2 -j $(( $i < $MAX_JOBS ? $i : $MAX_JOBS )) --rnd-seed $RANDOM &> /dev/null) "
        if [ $? -eq 0 ]; then
            break
        fi
    done
    
    printf "$n,"
    python print_graph_info.py data.csv
    rm data.csv
done
