#!/bin/bash

frac_train=('4' '8' '16' '32' '64' '128' '256' '512')
dev=0

##create an array of weight decay values

##loop through the array and run the training script for each value

for i in "${frac_train[@]}"
do
    echo "Batch size: $i"
    python train_grokk.py --bsize $i --device $dev > /dev/null 2> /dev/null &
    dev=$((dev+1))
done

echo "Done!"
