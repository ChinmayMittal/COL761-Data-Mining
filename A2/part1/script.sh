#!/bin/bash

ALGO=$1
MINSUP=$2

if [[ "$ALGO" == "fsg" ]] ;
then
    ./fsg data/fsg.txt_graph --support=$2
elif [[ "$ALGO" == "gspan" ]] ;
then
    ./gSpan-64 -f data/gspan.txt_graph -s 0.$2 -o
fi