#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>


struct keyComparator {
    bool operator()(const int a, const int b) const {
        return a > b;
    }
};


void decompress(std::string decompressed_file_path, std::string compressed_file_path)
{
    std::string line; int num ;
    std::map<int, std::set<int>, keyComparator> conversion_dictionary;
    std::ifstream compressed_file(compressed_file_path);

    if (!compressed_file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }

    std::ofstream outFile; 
    outFile.open(decompressed_file_path, std::ofstream::out | std::ofstream::trunc);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return ; // Return an error code
    }


    bool dictionary_started = false;
    while(std::getline(compressed_file, line))
    {
        if(dictionary_started)
        {
            int key;
            std::istringstream iss(line);
            iss >> key ;;
            while(iss >> num)
            {
                conversion_dictionary[key].insert(num);
            }
        }
        
        if(!dictionary_started and line.length() == 0)
        {
            dictionary_started = true;
        }
    }
    std::map<int, std::set<int>, keyComparator> new_conversion_dictionary;
    for(auto const &pr : conversion_dictionary)
    {
        for(auto ele : pr.second)
        {
            if (ele < 0)
            {
                for(auto element : new_conversion_dictionary[ele])
                {
                    new_conversion_dictionary[pr.first].insert(element);
                }
            }else{
                new_conversion_dictionary[pr.first].insert(ele);
            }
        }
    }
    conversion_dictionary.clear();

    compressed_file.clear();
    compressed_file.seekg(0, std::ios::beg);

    while(std::getline(compressed_file, line))
    {
        if(line.length() == 0)
        {
            break;
        }
        std::istringstream iss(line);
        while(iss>>num)
        {
            if(num > 0)
            {
                outFile << num << " ";
            }else{
                for(auto ele : new_conversion_dictionary[num])
                    outFile << ele << " ";
            }
        }
        outFile << "\n" ;
    }

    compressed_file.close();
    outFile.close();
}

int main(int argc, const char *argv[])
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string compressed_file_path = argv[1];
    std::string decompressed_file_path = argv[2];
    
    decompress(decompressed_file_path, compressed_file_path);
    
    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the duration in seconds
    std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;

    return EXIT_SUCCESS;
}