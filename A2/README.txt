(i)HW2_CS1200380
   |---part1
        |---data
            |---running_time_plot.png (running time vs support plot)
            |---time_output.txt (running times of mining algos)
            |---yeast (sample dataset)
        |---fsg
            |--fsg (executable for FSG algo on linux)
        |---gaston-1.1
            |---gaston (executable for gaston algo on linux)
            |...other source code
        |---gspan
            |---gSpan-64 (executable for gSpan algo on linux)
        |---format_change.cpp (source code to convert graph input file into required format)
        |---Makefile (build script for compiling format_change.cpp)
        |---plot.py (script for creating run time plots)
        |---run.sh (main shell file for running part 1)
   |---part2
        |---analysis_graph.py (script to make elbow plot)
        |---clustering.cpp (script to perform k-means clustering)
        |---elbow_plot.sh ( generic script to generate elbow plot which takes 3 arguments, dataset, dimension, and plot name)
        |---generateDataset_d_dim_hpc_compiled (executable file to generate dataset)
   |---CS1200380.pdf (report)
   |---CS1200380_install.sh (bash script to clone github repo)
   |---README.txt

(ii) Team Members-
    Chinmay Mittal, 2020CS10336
    Shah Parth Urveshkumar, 2020CS10380
    Shubh Goel, 2020EE10672

(iii) Instructions on how to execute our code
    cd HW2_CS1200380
    a) part1
        1) cd ./part1
        2) ./run.sh <dataset> (for example, ./run.sh ./data/yeast)
        running_time_plot.png will be created in the same dir.
    b) part2
        1)cd ./part2
        2)./generateDataset_d_dim_hpc_compiled <RollNo> <dimension> #Make sure the file has execute permissions
        3)./elbow_plot.sh <dataset> <dimension> q3_<dimension>_<RollNo>.png
        q3_<dimension>_<RollNo>.png will be created in the specified path
    c) part3
        No source code. Dendogram, algorithm and time complexity analysis are in the report

(iv)
    Chinmay Mittal - 33
    Shah Parth Urveshkumar - 33
    Shubh Goel - 33


