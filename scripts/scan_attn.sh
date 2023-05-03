#!/bin/bash

wd=('2' '4' '8' '16' '32' '64')
dev=0

##create an array of weight decay values

##loop through the array and run the training script for each value

for i in "${wd[@]}"
do
    python train_grokk.py --attn_dim $i --device $dev > /dev/null 2> /dev/null &
    dev=$((dev+1))
done

echo "Done!"
