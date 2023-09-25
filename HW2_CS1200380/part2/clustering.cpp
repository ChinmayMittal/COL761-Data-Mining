#include <bits/stdc++.h>
#include <algorithm>
#include <random>

using namespace std;

int main(int argc, char* argv[])
{
    // Read all the points to a dims x n array
    int dims = atoi(argv[2]);
    string filename = argv[1];
    ifstream data_file;
    data_file.open(filename);
    vector<double> data[dims];
    vector<int> index_for_random;
    string line;
    int n = 0;
    cout << "Data:\n";
    while(getline(data_file, line))
    {
        istringstream iss(line);
        for(int i = 0; i < dims; i++)
        {
            double x;
            iss >> x;
            data[i].push_back(x);
            cout << data[i][n] << " ";
        }
        cout << "\n";
        n++;
    }
    
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen;
    

    int max_k = atoi(argv[3]);
    int avg_parameter = atoi(argv[4]);
    ofstream out_file;
    out_file.open("clustering_values.csv", std::ios::out | std::ios::trunc);
    out_file << "distance" << "," << "dimensions" << "," <<"k"<< "\n";
    for(int k = 1;k<=max_k;k++){
        double medians[dims][k];
        double new_medians[dims][k];
        double distance[n][k];
        double cluster_index[n];
        double minimum_distance[n];
        double distance_total;
        vector<int> cluster_population(k, 1);
        
        
        
        for(int iter = 0;iter<avg_parameter;iter++){
            int num_iterations = 0;

            index_for_random.clear();
            for(int i = 0; i < n; i++)
            {
                index_for_random.push_back(i);
            }
            shuffle(index_for_random.begin(), index_for_random.end(), gen);
            cout << "Initial_medians:\n";
            for(int i = 0; i < k; i++)
            {
                for(int j = 0; j < dims; j++)
                {
                    new_medians[j][i] = data[j][index_for_random[i]];
                    cout << new_medians[j][i] << " " ;
                }
                cout << "\n";
            }
            bool completed = false;
            do
            {
                /* code */
                num_iterations++;
                for(int i = 0; i < k; i++)
                {
                    for(int j = 0; j < dims; j++)
                    {
                        medians[j][i] = new_medians[j][i];
                        new_medians[j][i] = 0;
                    }
                    cluster_population[i] = 0;
                }
                distance_total = 0;

                //for all the examples claculate the distance
                for(int i = 0; i < n; i++)
                {
                    int arg_min = -1;
                    double min_distance = INT_MAX;
                    //for a given example calculate the distance from the current median
                    for(int j = 0; j < k; j++)
                    {
                        distance[i][j] = 0;
                        //for a given median calculate the distance in all the dimentions
                        for(int z = 0; z < dims; z++)
                        {
                            distance[i][j] += pow(medians[z][j] - data[z][i], 2);
                        }
                        if(distance[i][j] < min_distance)
                        {
                            min_distance = distance[i][j];
                            arg_min = j;
                        }
                    }
                    //calculate the cluster for this example
                    cluster_index[i] = arg_min;
                    cluster_population[arg_min] += 1;

                    //add the current example for finding the median of its current cluster
                    for(int z = 0; z < dims; z++)
                    {
                        new_medians[z][arg_min] += data[z][i];
                    }
                    minimum_distance[i] = pow(min_distance, 0.5);
                    distance_total += min_distance;
                }

                //now take the average over all the points added to the new median
                for(int j = 0; j < k; j++)
                {
                    for(int z = 0; z < dims; z++)
                    {
                        if(cluster_population[j] == 0)
                            new_medians[z][j] = medians[z][j];
                        else
                            new_medians[z][j] /= cluster_population[j];
                    }
                }

                //check if the new medians overlap with the past medians if true then terminate
                completed = true;
                for(int i = 0; i < k; i++)
                {
                    for(int j = 0; j < dims; j++)
                    {
                        if(abs(medians[j][i] - new_medians[j][i]) > 1e-7)
                            completed = false;
                    }
                }

                cout << "New medians: \n";
                for(int i = 0; i < k; i++)
                {
                    for(int j = 0; j < dims; j++)
                    {
                        cout << new_medians[j][i] << " ";
                    }
                    cout << "\n";
                    cout << cluster_population[i] << "\n";
                }
                cout << "Iteration number: " << num_iterations << endl;
            } while (!completed);
            
            // ofstream out_file;
            // out_file.open("clustering_values.csv", ios::app);
            out_file << distance_total/n << "," << dims << "," << k << "\n";
        }
    }
    out_file.close();
}