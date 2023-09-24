#!/bin/bash

g++ clustering.cpp
for dim in {4..7}
do
    for clus in {1..15}
    do
        for runs in {1..100}
        do
            ./a.out CS1200380_generated_dataset_${dim}D.dat ${dim} ${clus}
        done
    done
done