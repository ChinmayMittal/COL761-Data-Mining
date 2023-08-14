#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "fptree.h"


void test_1()
{
    std::string file_path = "./test.dat";

    const uint64_t minimum_support_threshold = 2;

    const FpTree fptree{ file_path, minimum_support_threshold };

    std::set<Pattern> frequent_patterns = mine_fptree(fptree);

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
    test_1();
    std::cout << "All tests passed!" << std::endl;

    return EXIT_SUCCESS;
}