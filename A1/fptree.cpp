#include <algorithm>
#include <cstdint>
#include <utility>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>

#include "fptree.h"

FpNode::FpNode(const Item& item, const std::shared_ptr<FpNode>& parent):
    item(item),
    frequency(1),
    next_node_in_ht(nullptr),
    parent(parent),
    children()
{

}

FpTree::FpTree(const std::vector<Transaction>& transactions, uint64_t minimum_support_threshold):
    root(std::make_shared<FpNode>(Item{}, nullptr)),
    header_table(),
    minimum_support_threshold(minimum_support_threshold)
{
    // scan the transactions counting the frequency of each item
    std::map<Item, uint64_t> frequency_by_item;
    for ( const Transaction& transaction : transactions ) {
        for ( const Item& item : transaction ) {
            ++frequency_by_item[item];
        }
    }

    // keep only items which have a frequency greater or equal than the minimum support threshold
    for ( auto it = frequency_by_item.cbegin(); it != frequency_by_item.cend(); ) {
        const uint64_t item_frequency = (*it).second;
        if ( item_frequency < minimum_support_threshold ) { frequency_by_item.erase( it++ ); }
        else { ++it; }
    }

    // order items by decreasing frequency
    struct frequency_comparator
    {
        bool operator()(const std::pair<Item, uint64_t> &lhs, const std::pair<Item, uint64_t> &rhs) const
        {
            return std::tie(lhs.second, lhs.first) > std::tie(rhs.second, rhs.first);
        }
    };
    std::set<std::pair<Item, uint64_t>, frequency_comparator> items_ordered_by_frequency(frequency_by_item.cbegin(), frequency_by_item.cend());

    // start tree construction

    // scan the transactions again
    for ( const Transaction& transaction : transactions ) {
        auto curr_fpnode = root;

        // select and sort the frequent items in transaction according to the order of items_ordered_by_frequency
        for ( const auto& pair : items_ordered_by_frequency ) {
            const Item& item = pair.first;

            // check if item is contained in the current transaction
            if ( std::find( transaction.cbegin(), transaction.cend(), item ) != transaction.cend() ) {
                // insert item in the tree

                // check if curr_fpnode has a child curr_fpnode_child such that curr_fpnode_child.item = item
                const auto it = std::find_if(
                    curr_fpnode->children.cbegin(), curr_fpnode->children.cend(),  [item](const std::shared_ptr<FpNode>& fpnode) {
                        return fpnode->item == item;
                } );
                if ( it == curr_fpnode->children.cend() ) {
                    // the child doesn't exist, create a new node
                    const auto curr_fpnode_new_child = std::make_shared<FpNode>( item, curr_fpnode );

                    // add the new node to the tree
                    curr_fpnode->children.push_back( curr_fpnode_new_child );

                    // update the node-link structure
                    if ( header_table.count( curr_fpnode_new_child->item ) ) {
                        auto prev_fpnode = header_table[curr_fpnode_new_child->item];
                        while ( prev_fpnode-> next_node_in_ht ) { prev_fpnode = prev_fpnode->next_node_in_ht; }
                        prev_fpnode->next_node_in_ht = curr_fpnode_new_child;
                    }
                    else {
                        header_table[curr_fpnode_new_child->item] = curr_fpnode_new_child;
                    }

                    // advance to the next node of the current transaction
                    curr_fpnode = curr_fpnode_new_child;
                }
                else {
                    // the child exist, increment its frequency
                    auto curr_fpnode_child = *it;
                    ++curr_fpnode_child->frequency;

                    // advance to the next node of the current transaction
                    curr_fpnode = curr_fpnode_child;
                }
            }
        }
    }
}

FpTree::FpTree(const std::string& file_path, uint64_t minimum_support_threshold):
    root(std::make_shared<FpNode>(Item{}, nullptr)),
    header_table(),
    minimum_support_threshold(minimum_support_threshold)
{

    std::ifstream input_file(file_path);

    if (!input_file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }

    // scan all the transactions to determine the frequencies of all elements
    std::map<Item, uint64_t> item_frequencies;
    std::string line;
    while (std::getline(input_file, line)) {
        // process transactions
        int num;
        std::istringstream iss(line);

        while (iss >> num) {
            Item item{num};
            item_frequencies[item] ++;
        }
    }

    input_file.close();




    // keep only items with frequency greater than or equal to the minimum supported threshold
    for(auto item_frequencies_it = item_frequencies.cbegin(); item_frequencies_it != item_frequencies.cend() ;)
    {
        const uint64_t item_frequency = (*item_frequencies_it).second;
        if( item_frequency < minimum_support_threshold )
        {
            item_frequencies.erase(item_frequencies_it++);
        }else{
             ++item_frequencies_it;
        }
    }


    // order items by decreasing frequency
    struct frequency_comparator
    {
        bool operator() (const std::pair<Item, uint64_t> &lhs, const std::pair<Item, uint64_t> &rhs) const
        {
            return std::tie(lhs.second, lhs.first) > std::tie(rhs.second, rhs.first);
        }
    };
    // provide custom comparator to set
    std::set<std::pair<Item, uint64_t>, frequency_comparator> items_ordered_by_frequency(item_frequencies.cbegin(), item_frequencies.cend());

    // scan the transactions again
    std::ifstream input_f(file_path);

    if (!input_f.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return ;
    }

    while (std::getline(input_f, line)) {
        // generate transaction row by row
        int num;
        std::istringstream iss(line);
        Transaction transaction;
        while (iss >> num) {
            Item item{num};
            transaction.push_back(item);
        }

        // add the transaction to the tree
        auto curr_fpnode = root;

        for(const auto& item_frequency_pair : items_ordered_by_frequency)
        {
            const Item& item = item_frequency_pair.first;

            if(std::find(transaction.cbegin(), transaction.cend(), item) != transaction.cend())
            {
                // this high frequency item exists in the transaction and should be added to the tree

                const auto it = std::find_if(curr_fpnode->children.cbegin(), curr_fpnode->children.cend(), [item](const std::shared_ptr<FpNode>& fpnode){
                    return item == fpnode -> item;
                });

                if (it == curr_fpnode->children.cend())
                {
                    // need to create a node for this item as such a child doesn't exist
                    const auto new_fp_node_child = std::make_shared<FpNode>(item, curr_fpnode);

                    curr_fpnode->children.push_back(new_fp_node_child);

                    // update header table structure
                    if( header_table.count(item))
                    {
                        auto fp_node = header_table[item];
                        while(fp_node->next_node_in_ht)
                        {
                            fp_node = fp_node -> next_node_in_ht;
                        }
                        fp_node->next_node_in_ht = new_fp_node_child;
                    }else{
                        header_table[item] = new_fp_node_child;
                    }

                    // advance pointer down the tre
                    curr_fpnode = new_fp_node_child;
                }else{
                    // child exists
                    auto fp_node_child = *it;
                    ++fp_node_child->frequency;

                    //advance pointer down the tree
                    curr_fpnode = fp_node_child;
                }

            }
        }
    }

    input_f.close();

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
    return containts_single_path(fpnode->children.front());
}

bool containts_single_path(const FpTree& fptree)
{
    return fptree.empty() || containts_single_path(fptree.root);
}

std::set<Pattern> mine_fptree(const FpTree& fptree)
{
    if (fptree.empty()){ return {};}

    if (containts_single_path(fptree))
    {
        // generate all possible combinationso of items in the trees

        // eg. if tree is A->B->C then we will generate A, B, C, AB, BC, AC, ABC 
        std::set<Pattern> single_path_patterns;

        // for each node in tree
        auto fpnode = fptree.root->children.front();
        while (fpnode)
        {
            const Item& item = fpnode -> item ;
            const uint64_t frequency = fpnode -> frequency ;

            // add a pattern formed by only the current node
            // this will be frequent because we removed infrequent single items from the 
            Pattern new_pattern{{item}, frequency};
            single_path_patterns.insert(new_pattern);

            // create a new pattern by adding the item of the current node to all the previously generated patterns

            for( const Pattern& pattern : single_path_patterns )
            {
                Pattern new_Pattern{pattern};
                new_pattern.first.insert(item);
                new_pattern.second = frequency;
                single_path_patterns.insert(new_pattern);
            }
            if (fpnode->children.size())
            {
                fpnode = fpnode->children.front();
            }else{
                fpnode = nullptr ;
            }
        }
        return single_path_patterns;
    }
    else{

        // generaate conditional fptrees for each different item in the fptree

        std::set<Pattern> multi_path_patterns ;

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

            //generate teh transactions from the conditional pattern base
            std::vector<Transaction> conditional_fptree_transactions;
            for (const TransformedPrefixPath &transformed_prefix_path : conditional_pattern_base)
            {
                const std::vector<Item>& transformed_prefix_path_items = transformed_prefix_path.first;
                const uint64_t transformed_prefix_path_frequency = transformed_prefix_path.second;

                Transaction transaction = transformed_prefix_path_items;

                // add the same transaction frequncy number of times to the conditional tree
                for( auto i = 0 ; i < transformed_prefix_path_frequency ; ++i )
                {
                    conditional_fptree_transactions.push_back(transaction);
                }
            }

            // build the conditional fptree using the generated condition transactions
            const FpTree conditional_fptree( conditional_fptree_transactions, fptree.minimum_support_threshold);

            // this is a recursive function call
            // gets the frequent patters in the conditional FPTree 
            std::set<Pattern> conditional_patterns = mine_fptree( conditional_fptree ); // recursive function
        
            // construct patterns relative to the current item using both the current item and the conditional patterns
            std::set<Pattern> curr_item_patterns;

            // the first pattern is made only by the current item
            // compute the frequency of this pattern by summing the frequency of the nodes which have the same item (follow the node links)
            uint64_t curr_item_frequency = 0; // will represent the total frequency of the current item 
            auto fpnode = pair.second; // first node in the header_table of the first item
            while ( fpnode ) {
                curr_item_frequency += fpnode->frequency;
                fpnode = fpnode->next_node_in_ht;
            }
            // add the pattern as a result
            Pattern pattern{ { curr_item }, curr_item_frequency }; // this is a frequent pattern since we only considered frequent items 
            curr_item_patterns.insert( pattern );

            // other patterns are generated recursively by adding the current item to the recursively generated patterns
            for (const Pattern& pattern : conditional_patterns)
            {
                Pattern new_pattern{pattern};
                new_pattern.first.insert(curr_item);
                new_pattern.second = pattern.second;
                curr_item_patterns.insert({new_pattern});
            }

            // join the patterns generated by the current item with all the other items of the fptree
            multi_path_patterns.insert( curr_item_patterns.cbegin(), curr_item_patterns.cend() );
        }

        return multi_path_patterns;
    }
}