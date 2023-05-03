#!/bin/bash

frac_train=('0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8')
dev=0

##create an array of weight decay values

##loop through the array and run the training script for each value

for i in "${frac_train[@]}"
do
    echo "Weight decay: $i"
    python train_grokk.py --frac_train $i --device $dev > /dev/null 2> /dev/null &
    dev=$((dev+1))
done

echo "Done!"
