#!/bin/bash

# Check if a filename was provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Get the filename from the command line argument
filename="$1"

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found: $filename"
    exit 1
fi

# Count the number of '#' characters in the file and store it in a variable
count=$(grep -o "#" "$filename" | wc -l)

# Print the result
echo "Number of Graphs in $filename: $count"

# Run 'make' to compile the 'format_change' executable
make

# Check if 'make' was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

./format_change "$filename" fsg
./format_change "$filename" gspan
./format_change "$filename" gaston

# Define the output file for 'format_change'
fsg_output_file="fsg_output.txt"
gspan_output_file="gspan_output.txt"
gaston_output_file="gaston_output.txt"
time_output_file="time_output.txt"

support_values=("95" "90" "80" "70" "60" "50" "35" "25" "10" "5")  # Add more values as needed

for support_fsg in "${support_values[@]}"; do
    support_gspan=$(awk "BEGIN { printf \"%.2f\", $support_fsg / 100 }")
    support_gaston=$(awk "BEGIN { printf \"%.2f\", $support_gspan * $count }")
    echo "Support $support_fsg" >> "$time_output_file"
    echo -e "\nfsg: " >> "$time_output_file"
    { time  ./fsg/fsg -s "$support_fsg" "$filename-fsg"  >> "$fsg_output_file"; } 2>> "$time_output_file"
    echo -e "\ngspan: " >> "$time_output_file"
    { time  ./gspan/gSpan-64 -s "$support_gspan" -o -f "$filename-gspan"  >> "$gspan_output_file"; } 2>> "$time_output_file"
    echo -e "\ngaston: " >> "$time_output_file"
    { time  ./gaston-1.1/gaston "$support_gaston" "$filename-gaston"  >> "$gaston_output_file"; } 2>> "$time_output_file"
done