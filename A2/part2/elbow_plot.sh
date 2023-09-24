#!/bin/bash
# module load compiler/gcc/9.1.0
# Check if the required number of arguments is provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <dataset> <dimension> <output_image>"
    exit 1
fi

# Store the operation and arguments from command-line arguments
dataset="$1"
dimension="$2"
output_image="$3"

#3rd argument is the maximum value of k
#4th argument is the averaging parameter
./clustering ${dataset} ${dimension} 15 40

python3 analysis_graph.py --output_img ${output_image} --dim ${dimension}



