#!/bin/bash

# Check if the required number of arguments is provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <C/D> <input_file> <output_file>"
    exit 1
fi

# Store the operation and arguments from command-line arguments
operation="$1"
input_file="$2"
output_file="$3"

# Check if the operation is "C" (compression) or "D" (decompression)
if [ "$operation" == "C" ]; then
    # Execute the compressed.cpp program with the provided arguments
    ./main "$input_file" "$output_file"
elif [ "$operation" == "D" ]; then
    # Execute the decompressed.cpp program with the provided arguments
    ./decompress "$input_file" "$output_file"
else
    echo "Unknown operation: $operation"
    exit 1
fi
