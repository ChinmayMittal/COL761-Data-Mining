#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <bits/stdc++.h>

#include "fptree.h"

using namespace std;

struct Pattern_comparator
{
    bool operator ()(Pattern a, Pattern b) const
    {
        return (a.first.size() * (a.second - 1 )) > (b.first.size() * (b.second-1));
    }
};

void data_compression(std::string file_path, std::string compressed_file_path)
{

    std::vector<float> support_thresholds{1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.04, 0.01};
    std::map<std::set<int>, int> compression_dictionary ;
    std::string line ; int num ;
    int total_initial_terms = 0;
    int final_items = 0;
    int replacement_value = -1;
    std::string current_file = file_path;

    bool stop = false;

    // for(int iter = 0 ; iter < support_thresholds.size(); iter++)
    float iter = 1.14;
    float iter_diff = 0.1;
    long prev_increase[2] = {INT32_MAX, INT32_MAX};
    int past_frequent_patterns[2] = {0, 0};
    float huris1 = 7.0, huris2 = 1.3;
    while(!stop)
    {
        prev_increase[1] = prev_increase[0];
        prev_increase[0] = past_frequent_patterns[0] - past_frequent_patterns[1] + 1;
        past_frequent_patterns[1] = past_frequent_patterns[0];
        if(prev_increase[0]/prev_increase[1] > huris1)
        {
            iter_diff = iter_diff*0.7;
        }
        else if(prev_increase[0]/prev_increase[1] < huris2)
        {
            iter_diff = iter_diff*1.4;
        }
        iter_diff = min(1.0*iter_diff, iter*0.7);
        iter -= iter_diff;


        final_items = 0;
        const FpTree fptree{current_file, iter};



        auto start = std::chrono::system_clock::now();
        Time_check t1;
        t1.start_mining = &start;
        t1.stop_execution = false;
        std::set<Pattern> frequent_patterns = mine_fptree(fptree, t1);
        if(t1.stop_execution || iter <= 0.001)
        {
            stop = true;
        }
        std::cout << "Patterns Mined: " << frequent_patterns.size() << std::endl;
        past_frequent_patterns[0] = frequent_patterns.size();

        // for(auto pattern: frequent_patterns)
        // {
        //     for(auto ele : pattern.first)
        //     {
        //         std::cout << ele << ", " ;
        //     }
        //     std::cout << "->" << pattern.second << "\n"; 
        // }

        std::ifstream input_file(current_file);
        std::ofstream outFile; // Declare a file stream object
        if (!input_file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return ;
        }

        outFile.open( (current_file == file_path || current_file == "output-1.dat" ) ? "output.dat" : "output-1.dat" , std::ofstream::out | std::ofstream::trunc);
        if (!outFile.is_open()) {
            std::cerr << "Error opening file." << std::endl;
            return ; // Return an error code
        }


        while (std::getline(input_file, line)) {
            // process transactions
            std::istringstream iss(line);
            std::set<int> transaction ;

            while (iss >> num) {
                transaction.insert(num);
                if(iter==0) total_initial_terms ++ ;
            }

            // std::set<Pattern, Pattern_comparator> sorted_frequent_patterns(frequent_patterns.begin(), frequent_patterns.end());
            if(!stop)
            {
            for(auto const &pattern : frequent_patterns) // define order of processing transactions
            {
                if(pattern.first.size()>1 and pattern.second > 1)
                {     
                    if (std::includes(transaction.begin(), transaction.end(), pattern.first.begin(), pattern.first.end()))
                    {
                        if (compression_dictionary.count(pattern.first) == 0)
                        {
                            compression_dictionary[pattern.first] = replacement_value -- ;
                        }
                        for(const auto ele : pattern.first)
                        {
                            transaction.erase(ele);
                        }
                        transaction.insert(compression_dictionary[pattern.first]);
                    }
                }
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-*t1.start_mining;
                if(elapsed_seconds.count() > 30.0)
                {
                    stop = true;
                    break;
                }
            }
            }

            for(auto ele : transaction)
            {
                outFile << ele << " " ;
                final_items ++ ;
            }
            outFile << "\n";
        }
        input_file.close();
        outFile.close();
    
        current_file = (current_file == file_path || current_file == "output-1.dat") ? "output.dat" : "output-1.dat" ;
    }

    std::ofstream outFile; // Declare a file stream object
    outFile.open( current_file, std::ofstream::out | std::ofstream::app);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return ; // Return an error code
    }
    outFile << "\n" ;
    for(auto pair : compression_dictionary)
    {
        outFile << pair.second << " ";
        final_items ++ ;
        for(auto ele : pair.first)
        {
            outFile << ele << " " ;
            final_items ++ ;
        }
        outFile << "\n";
    }

    // Rename the file
    int result = std::rename(current_file.c_str(), compressed_file_path.c_str());
    result = std::remove( current_file == "output.dat" ? "output-1.dat" : "output.dat");
    std::cout << "Initial Items: " << total_initial_terms  << "\n";
    std::cout << "Final Items: " << final_items << "\n";
    std::cout << "Compression Ratio: " << float(final_items)/total_initial_terms*100  << "\n";
}

int main(int argc, const char *argv[])
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string file_path = argv[1];
    std::string compressed_file_path = argv[2];
    
    data_compression(file_path, compressed_file_path);
    
    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the duration in seconds
    std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;

    return EXIT_SUCCESS;
}