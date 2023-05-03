#!/bin/bash

wd=('0.3' '0.1' '0.03' '0.01' '0.003' '0.001' '0.0003' '0.0001' )
dev=0

##create an array of weight decay values

##loop through the array and run the training script for each value

for i in "${wd[@]}"
do
    echo "Weight decay: $i"
    python train_grokk.py --lr $i --device $dev > /dev/null 2> /dev/null &
    dev=$((dev+1))
done

echo "Done!"
