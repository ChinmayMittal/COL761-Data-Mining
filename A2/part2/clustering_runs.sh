#!/bin/bash

g++ clustering.cpp
# for dim in {5}
# do
for clus in {1..15}
do
    # for runs in {1..100}
    for runs in {1..10}
    do
        # ./a CS1200380_generated_dataset_${dim}D.dat ${dim} ${clus}
        ./a CS1200380_generated_dataset_5D.dat 5 ${clus}
    done
done
# done