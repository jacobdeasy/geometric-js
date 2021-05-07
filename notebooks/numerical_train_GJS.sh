#!/bin/bash
dset=$1
START=$2
END=$3
epochs=$4
lr=$5
loss=$6
mixt=$7
dims=$8
scale=2

for i in $(seq $START $END);
do
    y=`bc <<< "scale=2; $i/100"`
    python numerical-integrate.py --metrics $loss --num-mixture $mixt --dimensions $dims --seed $dset --lr $lr --epochs $epochs --A0 $y;
done
