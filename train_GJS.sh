#!/bin/bash
dset=$1
START=$2
END=$3
Epchs=$4
lr=$5
Lss=$6
scale=2

# for i in $(seq $START $END);
# do
#     python main.py 'constant-alpha' --dataset dset --epochs Epchs --loss Lss --GJS-A ${i} --rec-dist 'gaussian'
# done
for i in $(seq $START $END);
do
    y=`bc <<< "scale=2; $i/100"`
    python main.py 'constant-alpha' --dataset $dset --epochs $Epchs --loss $Lss --GJS-A $y --rec-dist 'gaussian';
done
