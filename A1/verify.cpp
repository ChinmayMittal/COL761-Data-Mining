#include<bits/stdc++.h>
using namespace std;

#define ll            long long int

vector<vector<ll>> get_transactions(string file1){
    vector<vector<ll>> transactions;
    ifstream file(file1,ios::in);
    if (file.good())
    {
        string str;
        while(getline(file, str)) 
        {
            vector<ll> transaction;
            istringstream ss(str);
            ll item;
            while(ss >> item)
            {
                transaction.push_back(item);
            }
            transactions.push_back(transaction);
        }
    }
    return transactions;
}

void find_error(string file1, string file2){
    auto transactions1 = get_transactions(file1);
    auto transactions2 = get_transactions(file2);

    if(transactions1.size() != transactions2.size()){
        cout<<"No. of transactions doesnt  match"<<"\n";
        return;
    }

    ll num = transactions1.size();

    for(ll i = 0;i<num;i++){
        auto transaction1 = transactions1[i];
        auto transaction2 = transactions2[i];
        if(transaction1.size() != transaction2.size()){
            cout<<"Size of " << i+1 << "transaction doesnt match"<<"\n";
            return;
        }
        sort(transaction1.begin(),transaction1.end());
        sort(transaction2.begin(),transaction2.end());
        ll tr_size = transaction1.size();
        for(ll j = 0;j<tr_size;j++){
            if(transaction1[j] != transaction2[j]){
                cout<<(j+1)<<"Item of "<<(i+1)<<" transaction doesnt match"<<"\n";
                return;
            }
        }
        
    }

    cout<<"No error!"<<"\n";
    
}


int main(){
    string file1 = "D_small.dat";
    string file2 = "decompressed.dat";
    find_error(file1,file2);
    return 0;
}



