#include <bits/stdc++.h>
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

struct Pattern_comparator
{
    bool operator ()(Pattern a, Pattern b) const
    {
        return (a.first.size() * (a.second - 1 )) > (b.first.size() * (b.second-1));
    }
};

void print_patterns(std::vector<Pattern>& patterns)
{
    std::cout << "Number of Patterns: " << patterns.size() << "\n" ;
    for(auto pattern: patterns)
    {
        for(auto ele : pattern.first)
        {
            std::cout << ele << ", " ;
        }
        std::cout << "->" << pattern.second << "\n"; 
    }
}
void data_compression(std::string file_path, std::string compressed_file_path)
{

    std::vector<float> support_thresholds{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001};
    // std::vector<float> support_thresholds{1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25};
    // std::vector<float> support_thresholds{0.45};
    std::map<std::set<int>, int> compression_dictionary ;
    std::vector<Transaction> transactions; // stores the current state of the transactions
    std::string line ; int num ;
    int total_initial_terms = 0; // total integers in the input file
    int final_items = 0; // total integers in the output file
    int replacement_value = -1; // current key for replacement

    // read transactions from file
    std::ifstream input_file(file_path);

    if (!input_file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }
    // store transactions from disk
    while(std::getline(input_file, line))
    {
        Transaction transaction;
        std::istringstream iss(line);
        while(iss >> num)
        {
            Item item{num};
            transaction.push_back(item);
            total_initial_terms ++ ;
        }
        transactions.push_back(transaction);
    }
    input_file.close();


    auto algo_start_time = std::chrono::high_resolution_clock::now();

    for(int iter = 0 ; iter < support_thresholds.size(); iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(start_time - algo_start_time);
        if (elapsed_time > std::chrono::seconds(1800))
        {
            std::cout << "Algorithm exceeded the maximum allowed execution time." << std::endl;
            break;
        }

        const FpTree fptree{transactions, support_thresholds[iter]};
        
        Time_check t1;
        t1.start_time = &start_time;
        t1.stop_execution = false;
        std::vector<Pattern> frequent_patterns = mine_fptree(fptree, t1);
        
        std::cout << "Patterns Mined: " << frequent_patterns.size() << std::endl;
        // print_patterns(frequent_patterns);

        if(!t1.stop_execution and (frequent_patterns.size() < std::min(fptree.total_transactions * 0.0003, 1000.0)))
        {
            std::cout << "Skipping support, too few frequent patterns \n";
            iter++;
            continue;
        }

        // size of pattern, id of pattern
        std::vector<std::pair<int, int>> pattern_sizes;
        for(int idx = 0 ; idx < frequent_patterns.size(); idx++)
        {
            pattern_sizes.push_back({(frequent_patterns[idx].first.size() - 1) * (frequent_patterns[idx].second-1), idx});
        }

        // sort patterns by reverse order of size
        std::sort(pattern_sizes.begin(), pattern_sizes.end(), std::greater<std::pair<int,int>>());

        for (int transaction_id = 0 ; transaction_id < transactions.size() ; transaction_id++ ) {
            // process transactions
            // std::istringstream iss(line);
            std::set<int> sorted_transaction(transactions[transaction_id].begin(), transactions[transaction_id].end()) ;

            // std::set<Pattern, Pattern_comparator> sorted_frequent_patterns(frequent_patterns.begin(), frequent_patterns.end());
            for(int idx = 0 ; idx < std::min(10000, (int)frequent_patterns.size()) ; idx ++) // define order of processing transactions
            {
                if(pattern_sizes[idx].first - 2 <= 0) break;
                Pattern &pattern = frequent_patterns[pattern_sizes[idx].second];
                if(pattern.first.size()>1 and pattern.second > 1)
                {     
                    if (std::includes(sorted_transaction.begin(), sorted_transaction.end(), pattern.first.begin(), pattern.first.end()))
                    {
                        if (compression_dictionary.count(pattern.first) == 0)
                        {
                            compression_dictionary[pattern.first] = replacement_value -- ;
                        }
                        for(const auto ele : pattern.first)
                        {
                            sorted_transaction.erase(ele);
                        }
                        sorted_transaction.insert(compression_dictionary[pattern.first]);
                    }
                }
            }
            // new compressed transaction
            transactions[transaction_id] = std::vector<Item>(sorted_transaction.begin(), sorted_transaction.end());
        }
    
        if (t1.stop_execution)
        {
            std::cout << "Terminating Compression, TLE\n";
            break;
        }
    }

    std::ofstream outFile; // Declare a file stream object
    outFile.open(compressed_file_path);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return ;
    }
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
    outFile << "\n" ;
    for(auto const &transaction : transactions)
    {
        for(auto const &item : transaction)
        {
            outFile << item << " ";
            final_items ++ ;
        }
        outFile << "\n";
    }

    outFile.close();

    // Print Statistics
    std::cout << "Initial Items: " << total_initial_terms  << "\n";
    std::cout << "Final Items: " << final_items << "\n";
    std::cout << "Amount of Compression: " << (100.0 - float(final_items)/total_initial_terms*100)  << "\n";
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