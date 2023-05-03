#!/bin/bash

wd=('16' '32' '64' '128' '256' '512' '1025' '2048')
dev=0

##create an array of weight decay values

##loop through the array and run the training script for each value

for i in "${wd[@]}"
do
    echo "Weight decay: $i"
    python train_grokk.py --hidden_dim $i --device $dev > /dev/null 2> /dev/null &
    dev=$((dev+1))
done

echo "Done!"
