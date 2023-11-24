#! /usr/bin/bash

models=(
    "Baseline"
    "GCN"
    "GAT"
    "GIN"
    "SAGE"
)

lr=(
    "0.001"
    "0.005"
    "0.01"
    "0.05"
    "0.1"
)

for hc in {4..7}
do
    for learning_rate in ${lr[@]}
    do
        for model in ${models[@]}
        do
            if [ $model == "Baseline" ]
            then
                python3 classify.py --hidden_channels=$(($hc**2)) --lr=$learning_rate --model=$model
            else
                for layers in {2..6}
                do
                    python3 classify.py --hidden_channels=$(($hc**2)) --lr=$learning_rate --model=$model --layers=$layers
                done
            fi
        done
    done
done