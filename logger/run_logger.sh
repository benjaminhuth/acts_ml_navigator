#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Usage: $0 <num_of_events>"
    exit 0
fi

N=0
MAX_JOBS=20
ODIR="../../data/logger/"

while true; do
    let N=N+1
    OUTPUT="$(./logger -n $1 --bf-values 0 0 2 -j $(( $1 < $MAX_JOBS ? $1 : $MAX_JOBS )) --rnd-seed $RANDOM --output-dir $ODIR &> /dev/null)"
    if [ $? -eq 0 ]; then
        echo "iteration $N successfull"
        break
    fi
    echo "iteration $N failed"
done

rm "$ODIR/timing.tsv"
