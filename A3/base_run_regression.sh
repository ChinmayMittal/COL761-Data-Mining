#!/bin/bash
exec > GNN-results-regression.txt
exec 2>&1
export PYTHONUNBUFFERED=1

models=(
    "Baseline"
    "GCN"
    "GAT"
    "GIN"
    "SAGE"
)

# models=(
#     "GCN"
# )

# lr=(
#     "0.001"
#     "0.005"
#     "0.01"
#     "0.05"
#     "0.1"
# )

lr=(
    "0.001"
)

layers=(1 2 4)

hidden_channels=(64 128)

for model in ${models[@]}
do
    for learning_rate in ${lr[@]}
    do
        for hc in "${hidden_channels[@]}"
        do
            if [ $model == "Baseline" ]
            then
                echo "Model: $model" "Hidden Channels: $hc" 
                python3 regression.py --hidden_channels=$(($hc)) --lr=$learning_rate --model=$model
            else
                for layer in "${layers[@]}"
                do
                    echo "Model: $model" "Hidden Channels: $hc" "Layers: $layer"
                    python3 regression.py --hidden_channels=$(($hc)) --lr=$learning_rate --model=$model --layers=$layer
                done
            fi
        done
    done
done