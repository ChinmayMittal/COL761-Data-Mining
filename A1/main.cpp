#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdio>

#include "fptree.h"


void data_compression(std::string file_path)
{

    std::vector<float> support_thresholds{1.0, 0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01};
    std::map<std::set<int>, int> compression_dictionary ;
    std::string line ; int num ;
    int total_initial_terms = 0;
    int final_items = 0;
    int replacement_value = -1;
    std::string current_file = file_path;

    for(int iter = 0 ; iter < support_thresholds.size(); iter++)
    {
        final_items = 0;
        const FpTree fptree{current_file, support_thresholds[iter]};
        std::set<Pattern> frequent_patterns = mine_fptree(fptree);
        std::cout << "Patterns Mined: " << frequent_patterns.size() << std::endl;

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

            for(auto const &pattern : frequent_patterns) // define order of processing transactions
            {
                if(pattern.first.size()>1 and pattern.second > 1)
                {
                    bool pattern_in_transaction = true;
                    for(auto const ele : pattern.first)
                    {
                        if(transaction.find(ele) == transaction.end())
                        {
                            pattern_in_transaction = false;
                        }
                    }      break;
                      
                    if (pattern_in_transaction)
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
    int result = std::rename(current_file.c_str(), "compressed.dat");
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
    
    data_compression(file_path);
    
    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the duration in seconds
    std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;

    return EXIT_SUCCESS;
}