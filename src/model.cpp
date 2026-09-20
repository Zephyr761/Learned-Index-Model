#include "model.h"

int BinarySearch(std::vector<std::string> data, std::string k)
{
    std::sort(data.begin(), data.end());
    int sz = data.size();
    int a = 0;
    int b = sz - 1;
    while(a < b)
    {
        if(k > data[(a + b) / 2]) a = (a + b) / 2 + 1;
        else b = (a + b) / 2;
    }
    return a;
}

void LinearModel::init(std::vector<std::string> dt)
{
    this->data.input = dt;
    std::sort(this->data.input.begin(), this->data.input.end());
    for(int i = 0; i < this->data.input.size(); ++i)
    {
        this->data.rank[i] = i;
        this->data.encoded[i] = 0;
        for(int j = 0; j < 8 && j < this->data.input[i].size(); ++j) this->data.encoded[i] += 256^(7-j)*this->data.input[i][j];
    }
    this->max_encoded = *std::max_element(this->data.encoded.begin(), this->data.encoded.end());
    this->min_encoded = *std::min_element(this->data.encoded.begin(), this->data.encoded.end());
    this->k = dt.size() - 1;
    this->b = 0;
    
}

int LinearModel::forward(std::string k)
{
    long long x = 0;
    for(int j = 0; j < 8 && j < k.size(); ++j) x += 256^(7 - j)*k[j];
    double xn = (x - this->min_encoded)*1.0 / (this->max_encoded - this->min_encoded);
    return int(this->k * xn + b);
}

