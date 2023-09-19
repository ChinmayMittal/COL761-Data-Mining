#include <bits/stdc++.h>

using namespace std;

struct graph{
    string graph_id;
    int num_nodes;
    vector<string> node_id;
    int num_edges;
    vector<pair<int, int>> edges;
    vector<string> edge_id;
};

int main(int argc, char* argv[])
{
    ifstream input_file;
    input_file.open("data/167.txt_graph");
    vector<graph*> graph_map;
    map<string, int> node_id_to_int;
    int node_count = 0;

    bool read_file = true;
    while(read_file)
    {
        string s;
        input_file >> s;
        if(s[0] != '#')
        {
            break;
        }
        graph *a = new graph;
        a->graph_id = s;
        input_file >> a->num_nodes;
        for(int i = 0; i < a->num_nodes; i++)
        {
            string c;
            input_file >> c;
            if(node_id_to_int.find(c) == node_id_to_int.end())
                node_id_to_int[c] = node_count++;
            a->node_id.push_back(c);
            
        }
        input_file >> a->num_edges;
        for(int i = 0; i < a->num_edges; i++)
        {
            int p1, p2;
            string p3;
            input_file >> p1 >> p2 >> p3;
            a->edges.push_back(make_pair(p1, p2));
            a->edge_id.push_back(p3);
        }
        // input_file >> s;
        graph_map.push_back(a);
    }
    cout << graph_map.size() << endl;


    string target_format = (argv[1]);

    if(target_format == "fsg")
    {
        ofstream out_file;
        out_file.open("data/fsg.txt_graph");
        for(auto a : graph_map)
        {
            out_file << "t " << a->graph_id << "\n";
            for(int i = 0; i < a->num_nodes; i++)
            {
                out_file << "v " << i << " " << a->node_id[i] << "\n";
            }
            for(int i = 0; i < a->num_edges; i++)
            {
                out_file << "e " << a->edges[i].first << " " << a->edges[i].second << " " << a->edge_id[i] << "\n";
            }
        }
    }

    if(target_format == "gspan")
    {
        ofstream out_file;
        out_file.open("data/gspan.txt_graph");
        int cnt = 0;
        for(auto a : graph_map)
        {
            out_file << "t # " << cnt << "\n";
            for(int i = 0; i < a->num_nodes; i++)
            {
                out_file << "v " << i << " " << node_id_to_int[a->node_id[i]] << "\n";
            }
            for(int i = 0; i < a->num_edges; i++)
            {
                out_file << "e " << a->edges[i].first << " " << a->edges[i].second << " " << a->edge_id[i] << "\n";
            }
            cnt++;
        }
    }




    for(auto a : graph_map)
    {
        delete a;
    }
}