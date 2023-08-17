#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "fptree.h"


void test_1(std::string file_path, float support_threshold)
{

    std::cout << "File Path " << file_path << ", Support Threshold " << (support_threshold*100) << "%" << std::endl;

    const FpTree fptree{file_path, support_threshold};

    std::cout << "Tree built " << std::endl ; 

    std::set<Pattern> frequent_patterns = mine_fptree(fptree);
    std::cout << frequent_patterns.size() << std::endl;
    for( const auto &pattern : frequent_patterns)
    {
        for(const auto &item : pattern.first)
        {
            std::cout << item << ", " ;
        }
        std::cout << "-- > " << pattern.second << " " << std::endl ;
    }

}

int main(int argc, const char *argv[])
{
    std::string file_path = argv[1];
    float support_threshold = std::stof(argv[2]);
    test_1(file_path, support_threshold);
    std::cout << "All tests passed!" << std::endl;

    return EXIT_SUCCESS;
}