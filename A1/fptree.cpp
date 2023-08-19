#include <algorithm>
#include <cstdint>
#include <utility>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>

#include "fptree.h"

void printProgressBar(int progress, int total, int barWidth = 50) {
    float percentage = static_cast<float>(progress) / total;
    int barProgress = static_cast<int>(barWidth * percentage);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < barProgress) {
            std::cout << "=";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << static_cast<int>(percentage * 100.0) << "%\r";
    std::cout.flush();
}


FpNode::FpNode(const Item& item, const std::shared_ptr<FpNode>& parent):
    item(item),
    frequency(1),
    next_node_in_ht(nullptr),
    parent(parent),
    children()
{

}

FpTree::FpTree(const std::vector<TransformedPrefixPath>& conditional_pattern_base, uint64_t minimum_support_threshold):
    root(std::make_shared<FpNode>(Item{}, nullptr)),
    header_table(),
    item_frequencies(),
    minimum_support_threshold(minimum_support_threshold)
{
    // TransformedPrefixPath is a path in the FP Tree corresponding to an item set and the frequency of that item set
    this->total_transactions = 0;
    this->total_items = 0;
    // scan the transactions counting the frequency of each item
    for ( const TransformedPrefixPath& tfp : conditional_pattern_base ) {
        this->total_transactions += tfp.second;
        for ( const Item& item : tfp.first ) {
            item_frequencies[item] += tfp.second;
            this->total_items += tfp.second;
        }
    }

    // keep only items which have a frequency greater or equal than the minimum support threshold
    for ( auto it = item_frequencies.cbegin(); it != item_frequencies.cend(); ) {
        const uint64_t item_frequency = (*it).second;
        if ( item_frequency < minimum_support_threshold ) { item_frequencies.erase( it++ ); }
        else { ++it; }
    }

    // start tree construction
    // scan the transactions again
    for ( const TransformedPrefixPath& tfp : conditional_pattern_base ) {

        Transaction transaction;
        // remove infrequent items from the transaction database
        for(const Item& item : tfp.first)
        {
            if(item_frequencies.count(item))
            {
                transaction.push_back(item);
            }
            
        }
        // sort transaction by frequency
        sort(transaction.begin(), transaction.end(), [this](Item a, Item b){
            return this->item_frequencies[a] > this->item_frequencies[b] ;
        });

        auto curr_fpnode = root;

        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        for ( const Item& item : transaction ) {
            // insert item in the tree
             // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
            if ( curr_fpnode->children.find(item) == curr_fpnode->children.cend() ) {
                // the child doesn't exist, create a new node
                const auto curr_fpnode_new_child = std::make_shared<FpNode>( item, curr_fpnode );
                curr_fpnode_new_child->frequency = tfp.second;
                // add the new node to the tree
                curr_fpnode->children[item] = ( curr_fpnode_new_child );

                // update the node-link structure
                if ( header_table.count( curr_fpnode_new_child->item ) ) {
                    last_node_in_header_table[curr_fpnode_new_child->item]->next_node_in_ht = curr_fpnode_new_child;
                    last_node_in_header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
                }
                else {
                    header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
                    last_node_in_header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
                }

                // advance to the next node of the current transaction
                curr_fpnode = curr_fpnode_new_child;
            }
            else {
                // the child exist, increment its frequency
                auto curr_fpnode_child = curr_fpnode->children[item];
                curr_fpnode_child->frequency += tfp.second;

                // advance to the next node of the current transaction
                curr_fpnode = curr_fpnode_child;
            }
            
        }
    }
}

FpTree::FpTree(const std::string& file_path, float minimum_support_threshold):
    root(std::make_shared<FpNode>(Item{}, nullptr)),
    item_frequencies(),
    header_table()
{
    std::ifstream input_file(file_path);

    if (!input_file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }

    // scan all the transactions to determine the frequencies of all elements
    std::string line;
    uint64_t total_transactions = 0;
    this->total_items=0;
    while (std::getline(input_file, line)) {
        // process transactions
        int num;
        std::istringstream iss(line);

        while (iss >> num) {
            Item item{num};
            item_frequencies[item] ++;
            this->total_items++ ;
        }
        total_transactions ++;
    }
    input_file.close();
    this->total_transactions = total_transactions;


    this->minimum_support_threshold = int(minimum_support_threshold*total_transactions);
    std::cout << "Support --> " << this->minimum_support_threshold << std::endl;
    // keep only items with frequency greater than or equal to the minimum supported threshold
    for(auto item_frequencies_it = item_frequencies.cbegin(); item_frequencies_it != item_frequencies.cend() ;)
    {
        const uint64_t item_frequency = (*item_frequencies_it).second;
        if( item_frequency < this->minimum_support_threshold )
        {
            item_frequencies.erase(item_frequencies_it++);
        }else{
             ++item_frequencies_it;
        }
    }

    // scan the transactions again
    std::ifstream input_f(file_path);

    if (!input_f.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }

    int transactions_processed = 0;
    while (std::getline(input_f, line)) {
        // generate transaction row by row
        int num;
        std::istringstream iss(line);
        Transaction transaction;
        while (iss >> num) {
            Item item{num};
            if (item_frequencies.count(item)) // only add frequent items
            {
                transaction.push_back(item);
            }
        }
        // printProgressBar(transactions_processed++, total_transactions, 100);
        // sort transaction by frequency
        sort(transaction.begin(), transaction.end(), [this](Item a, Item b){
            return this->item_frequencies[a] > this->item_frequencies[b] ;
        });

        // add the transaction to the tree
        auto curr_fpnode = root;

        for( const Item& item : transaction)
        {
            if (curr_fpnode->children.find(item) == curr_fpnode->children.cend())
            {
                // need to create a node for this item as such a child doesn't exist
                const auto new_fp_node_child = std::make_shared<FpNode>(item, curr_fpnode);

                curr_fpnode->children[item] = (new_fp_node_child);

                // update header table structure
                if( header_table.count(item))
                {
                    last_node_in_header_table[item]->next_node_in_ht = new_fp_node_child;
                    last_node_in_header_table[item] = new_fp_node_child;
                }else{
                    header_table[item] = new_fp_node_child;
                    last_node_in_header_table[item] = new_fp_node_child;
                }

                // advance pointer down the tre
                curr_fpnode = new_fp_node_child;
            }else{
                // child exists
                auto fp_node_child = curr_fpnode->children[item];
                ++fp_node_child->frequency;

                //advance pointer down the tree
                curr_fpnode = fp_node_child;
            }
        }

            
        
    }

    input_f.close();

}

FpTree::FpTree(const std::vector<Transaction>& transactions, float minimum_support_threshold) :
    root(std::make_shared<FpNode>( Item{}, nullptr ) ),
    header_table(),
    item_frequencies()
{
    // scan the transactions counting the frequency of each item
    this->total_transactions = transactions.size();
    this->total_items = 0;

    for ( const Transaction& transaction : transactions ) {
        for ( const Item& item : transaction ) {
            ++item_frequencies[item];
            this->total_items ++ ;
        }
    }

    this->minimum_support_threshold = int(minimum_support_threshold*(this->total_transactions));
    std::cout << "Support --> " << this->minimum_support_threshold << std::endl;

    // keep only items which have a frequency greater or equal than the minimum support threshold
    std::vector<Item> items_to_erase;
    for ( auto it = item_frequencies.cbegin(); it != item_frequencies.cend(); it++) {
        const uint64_t item_frequency = (*it).second;
        // if ( item_frequency < this->minimum_support_threshold ) { item_frequencies.erase( it++ ); }
        // else { ++it; }
        if(item_frequency < this->minimum_support_threshold) {items_to_erase.push_back((*it).first);}
    }

    for(const auto item : items_to_erase) {
        item_frequencies.erase(item);
    }

    // start tree construction

    // scan the transactions again
    for (Transaction transaction : transactions ) {
        // sort the transcation again by decreasing frequency
        Transaction pruned_transcation;
        for(auto ele : transaction)
        {
            if(item_frequencies.count(ele))
            {
                pruned_transcation.push_back(ele);
            }
        }
        sort(pruned_transcation.begin(), pruned_transcation.end(), [this](Item a, Item b)
        {
            return this->item_frequencies[a] > this->item_frequencies[b];
        });
        auto curr_fpnode = root;
        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        for (const Item& item : pruned_transcation ) {
            // insert item in the tree
            // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
            if ( curr_fpnode->children.find(item) == curr_fpnode->children.cend() ) {
                // the child doesn't exist, create a new node
                const auto new_fp_node_child = std::make_shared<FpNode>( item, curr_fpnode );

                // add the new node to the tree
                curr_fpnode->children[item] = (new_fp_node_child) ;

                // update the node-link structure
                if ( header_table.count(item)) {
                    last_node_in_header_table[item]->next_node_in_ht = new_fp_node_child;
                    last_node_in_header_table[item] = new_fp_node_child;
                }
                else {
                    header_table[item] = new_fp_node_child;
                    last_node_in_header_table[item] = new_fp_node_child;
                }

                // advance to the next node of the current transaction
                curr_fpnode = new_fp_node_child;
            }
            else {
                // the child exist, increment its frequency
                auto fp_node_child = curr_fpnode->children[item];
                ++fp_node_child->frequency;

                // advance to the next node of the current transaction
                curr_fpnode = fp_node_child;
            }
            
        }
    }
}

bool FpTree::empty() const
{
    return root->children.empty();
}

// these functions test if a tree is a single path A --> B --> C --> D --> E -- > -|
bool containts_single_path(const std::shared_ptr<FpNode>& fpnode)
{
    if (fpnode -> children.size() == 0)
    {
        return true;
    }
    if(fpnode -> children.size() > 1)
    {
        return false;
    }
    return containts_single_path((*(fpnode->children.begin())).second);
}

bool containts_single_path(const FpTree& fptree)
{
    return fptree.empty() || containts_single_path(fptree.root);
}

std::vector<Pattern> mine_fptree(const FpTree& fptree)
{
    if (fptree.empty()){ return {};}

    if (containts_single_path(fptree))
    {
        // generate all possible combinationso of items in the trees

        // eg. if tree is A->B->C then we will generate A, B, C, AB, BC, AC, ABC 
        std::vector<Pattern> single_path_patterns;

        // for each node in tree
        auto fpnode = (*(fptree.root->children.begin())).second;
        while (fpnode)
        {
            const Item& item = fpnode -> item ;
            const uint64_t frequency = fpnode -> frequency ;

            // add a pattern formed by only the current node
            // this will be frequent because we removed infrequent single items from the 
            Pattern single_item_pattern{{item}, frequency};

            // create a new pattern by adding the item of the current node to all the previously generated patterns
            int curr_size = single_path_patterns.size();
            for( int idx = 0 ; idx < curr_size ; idx ++  )
            {
                Pattern new_pattern = single_path_patterns[idx];
                new_pattern.first.insert(item);
                new_pattern.second = frequency;
                single_path_patterns.push_back(new_pattern);
            }
            single_path_patterns.push_back(single_item_pattern);
            if (fpnode->children.size())
            {
                fpnode = (*(fpnode->children.begin())).second;
            }else{
                fpnode = nullptr ;
            }
        }
        return single_path_patterns;
    }
    else{
        // generaate conditional fptrees for each different item in the fptree
        std::vector<Pattern> multi_path_patterns ;

        for (const auto & pair : fptree.header_table)
        {
            // pair is <Item, FpNode ---> FpNode --> FpNode --> NULL>
            const Item& curr_item = pair.first;

            // build the conditional fptree, conditoned on the current item
            // start generating the conditional pattern base
            std::vector<TransformedPrefixPath> conditional_pattern_base; /// TransformedPrefixPath is pair<vector<Item>, frequency>

            // for each node in the header table corresponding to the current item
            auto item_node = pair.second;
            while(item_node) // this loop iterates over all nodes of an item from the header table 
            {
                // each item in th transformed prefix path has the same frequency (the frequency of path_starting_fpnode)
                const uint64_t path_starting_fpnode_frequency = item_node->frequency;

                auto curr_path_fpnode = item_node->parent.lock(); // starting node of path
                // check if curr_path_fpnode is already the root of the fptree
                if(curr_path_fpnode->parent.lock())
                {
                    TransformedPrefixPath transformed_prefix_path{{}, path_starting_fpnode_frequency};
                    while(curr_path_fpnode->parent.lock())
                    {
                        transformed_prefix_path.first.push_back(curr_path_fpnode->item);
                        curr_path_fpnode = curr_path_fpnode->parent.lock(); // move up the path
                    }
                    conditional_pattern_base.push_back(transformed_prefix_path);
                }

                // advance to the next node in the header table
                item_node = item_node ->next_node_in_ht;
            }


            // build the conditional fptree using the generated condition transactions
            const FpTree conditional_fptree(conditional_pattern_base, fptree.minimum_support_threshold);
            // this is a recursive function call
            // gets the frequent patters in the conditional FPTree 
            std::vector<Pattern> conditional_patterns = mine_fptree(conditional_fptree); // recursive function
        
            // construct patterns relative to the current item using both the current item and the conditional patterns
            std::vector<Pattern> curr_item_patterns;

            // the first pattern is made only by the current item
            // compute the frequency of this pattern by summing the frequency of the nodes which have the same item (follow the node links)
            uint64_t curr_item_frequency = (*fptree.item_frequencies.find(curr_item)).second;
            // add the pattern as a result
            Pattern pattern{{ curr_item }, curr_item_frequency }; // this is a frequent pattern since we only considered frequent items 
            curr_item_patterns.push_back( pattern );

            // other patterns are generated recursively by adding the current item to the recursively generated patterns
            for (const Pattern& pattern : conditional_patterns)
            {
                Pattern new_pattern{pattern};
                new_pattern.first.insert(curr_item);
                new_pattern.second = pattern.second;
                curr_item_patterns.push_back({new_pattern});
            }

            // join the patterns generated by the current item with all the other items of the fptree
            multi_path_patterns.insert(multi_path_patterns.end(), curr_item_patterns.cbegin(), curr_item_patterns.cend() );
        }

        return multi_path_patterns;
    }
}