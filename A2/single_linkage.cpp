#include <bits/stdc++.h>

using namespace std;

struct data_point
{
    double x;
    double y;
};

struct cluster
{
    int id;
    double height;
    int cl1;
    int cl2;
    vector<data_point> data_points;
};

inline bool operator<(const cluster& lhs, const cluster& rhs)
{
  return lhs.id < rhs.id;
}

double dist(cluster cl1, cluster cl2){
    double min_dist = (double)LLONG_MAX;
    for(int i = 0;i<cl1.data_points.size();i++){
        for(int j = 0;j<cl2.data_points.size();j++){
            double curr = (cl1.data_points[i].x - cl2.data_points[j].x)*(cl1.data_points[i].x - cl2.data_points[j].x);
            curr+= (cl1.data_points[i].y - cl2.data_points[j].y)*(cl1.data_points[i].y - cl2.data_points[j].y);
            curr = sqrt(curr);
            min_dist = min(curr, min_dist);
        }
    }
    return min_dist;
}

cluster join(cluster cluster1, cluster cluster2, double distance, int curr_id){
    cluster new_cluster;
    new_cluster.id = curr_id;
    new_cluster.height = max(cluster1.height,cluster2.height) + distance;
    new_cluster.cl1 = cluster1.id;
    new_cluster.cl2 = cluster2.id;
    new_cluster.data_points = cluster1.data_points;
    for(auto &ele : cluster2.data_points){
        new_cluster.data_points.push_back(ele);
    }
    return new_cluster;
}

vector<cluster> single_linkage(set<cluster> &clusters){
    cout<<"In the function single_linkage"<<"\n";
    vector<cluster> total(clusters.begin(),clusters.end());
    int curr_id = clusters.size() + 1;
    while(clusters.size() > 1){
        double min_dist = (double)LLONG_MAX;
        cluster cl1,cl2;
        for(auto itr1 = clusters.begin();itr1!= clusters.end();itr1++){
            itr1++;
            auto temp = itr1;
            itr1--;
            for(auto itr2 = temp;itr2!=clusters.end();itr2++){
                double cluster_dist = dist((*itr1),(*itr2));
                if(cluster_dist < min_dist){
                    min_dist = cluster_dist;
                    cl1 = *itr1;
                    cl2 = *itr2;
                }
            }
        }
        cout<<"found new cluster: "<<curr_id<<"\n";
        clusters.erase(cl1);
        clusters.erase(cl2);

        cluster new_cluster = join(cl1,cl2,min_dist,curr_id);
        clusters.insert(new_cluster);
        total.push_back(new_cluster);
        curr_id++;
    }
    return total;
}

set<cluster> read_file(string file_name){
    ifstream input_file(file_name);

    if (!input_file.is_open()) {
        cout << "Failed to open the file." << "\n";
        return {};
    }
    set<cluster> init_clusters;
    int curr_id = 1;
    string line;
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
        cluster cl;
        cl.id = curr_id;
        cl.cl1 = curr_id;
        cl.cl2 = curr_id;
        cl.height = 0;
        cl.data_points.push_back(pt);
        init_clusters.insert(cl);
        // cout<<init_clusters.size()<<"\n";
        curr_id++;
    }
    input_file.close();

    return init_clusters;
}

int main(int argc, char const *argv[])
{
    string file_name = argv[1];

    auto init_clusters = read_file(file_name);
    // cout<<init_clusters.size()<<"\n";

    auto ans = single_linkage(init_clusters);

    for(auto ele : ans){
        cout<<"cluster id: "<<ele.id<<" cluster height: "<<ele.height<<" left cluster: "<<ele.cl1<<" right cluster: "<<ele.cl2;
        // cout<<" data points: ";
        // for(auto pt: ele.data_points){
        //     cout<<"("<<pt.x<<","<<pt.y<<") ";
        // }
        cout<<"\n";
    }
    
    return 0;
}


