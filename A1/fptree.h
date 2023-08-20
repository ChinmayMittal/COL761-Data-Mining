#ifndef FPTREE_HPP
#define FPTREE_HPP

#include<vector>
#include<map>
#include<memory>
#include<set>
#include<chrono>

using Item = int;
using Transaction = std::vector<Item>;
using TransformedPrefixPath = std::pair<std::vector<Item>, uint64_t>;
using Pattern = std::pair<std::set<Item>, uint64_t>;


class FpNode
{
    public:
        const Item item;
        uint64_t frequency;
        std::shared_ptr<FpNode> next_node_in_ht;
        std::weak_ptr<FpNode> parent;
        std::map<Item, std::shared_ptr<FpNode>> children;
        
        FpNode(const Item&, const std::shared_ptr<FpNode>&);
};


class FpTree
{
    public:
        std::shared_ptr<FpNode> root;
        std::map<Item, std::shared_ptr<FpNode>> header_table;
        std::map<Item, std::shared_ptr<FpNode>> last_node_in_header_table;
        std::map<Item, uint64_t> item_frequencies;
        uint64_t minimum_support_threshold;
        uint64_t total_transactions;
        uint64_t total_items;

        FpTree(const std::string&, float);
        FpTree(const std::vector<TransformedPrefixPath>& transactions, uint64_t);
        FpTree(const std::vector<Transaction>&, float);

        bool empty() const;
};

struct Time_check
{
    bool stop_execution = true;
    std::chrono::high_resolution_clock::time_point* start_time = NULL;
};

std::vector<Pattern> mine_fptree(const FpTree&, Time_check&);


#endif  // FPTREE_HPP