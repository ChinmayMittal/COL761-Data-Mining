#include <bits/stdc++.h>

using namespace std;

struct data_point
{
    int id;
    double x;
    double y;
};


vector<data_point> read_file(string file_name){
    ifstream input_file(file_name);

    if (!input_file.is_open()) {
        cout << "Failed to open the file." << "\n";
        return {};
    }
    
    int curr_id = 1;
    string line;
    vector<data_point> dpts;
    while(std::getline(input_file, line))
    {
        data_point pt;
        istringstream iss(line);
        double x,y;
        iss>>x;
        iss>>y;
        cout<<x<<" "<<y<<"\n";
        pt.x = x;
        pt.y = y;
        pt.id = curr_id;
        dpts.push_back(pt);
        // cout<<init_clusters.size()<<"\n";
        curr_id++;
    }
    input_file.close();

    return dpts;
}

vector<int> cluster_id(2001);
vector<int> cluster_size(2001,1);
vector<int> point_id(2001,1);
vector<double> height(2001,0);
vector<vector<int>> adj(2001);

int find(int id){
    while(id != point_id[id]){
        id = point_id[id];
    }
    return id;
}

void join(int id1, int id2){
    id1 = find(id1);
    id2 = find(id2);
    if(cluster_size[id1] < cluster_size[id2]) swap(id1,id2);
    cluster_size[id1] += cluster_size[id2];
    point_id[id2] = id1;
}

double dist(data_point a, data_point b){
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}

void single_linkage(vector<data_point> &dpts){
    int n = dpts.size();
    
    for(int i = 1;i<=n;i++){
        cluster_id[i] = i;
        point_id[i] = i;
    }

    vector<pair<double,pair<int,int>>> distances;

    //Calculate distances between each points
    for(int i = 0;i<n;i++){
        for(int j = i+1;j<n;j++){
            distances.push_back({dist(dpts[i],dpts[j]),{dpts[i].id,dpts[j].id}});
        }
    }

    sort(distances.begin(),distances.end());

    int new_cluster = n+1;
    
    //find(x) takes O(ln(n)) time (kruskal's algo)
    for(int i = 0;i<n*n;i++){
        int pt1 = distances[i].second.first;
        int pt2 = distances[i].second.second;
        if(find(pt1) == find(pt2)) continue;;
        cout<<pt1<<" "<<pt2<<"\n";
        height[new_cluster] = distances[i].first;
        adj[new_cluster].push_back(cluster_id[find(pt1)]);
        adj[new_cluster].push_back(cluster_id[find(pt2)]);
        join(pt1,pt2);  //O(ln(n));
        cluster_id[find(pt1)] = new_cluster;
        new_cluster++;
        if(new_cluster == 2*n) break;
    }

}

int main(int argc, char const *argv[])
{
    string file_name = argv[1];

    auto data_points = read_file(file_name);
    // cout<<init_clusters.size()<<"\n";

    single_linkage(data_points);

    for(int i = 1; i<= 2*data_points.size() - 1; i++){
        cout<<i<<" ";
        for(auto ele: adj[i]){
            cout<<ele<<" ";
        }
        cout<<"height: "<<height[i]<<"\n";
    }
    
    return 0;
}


