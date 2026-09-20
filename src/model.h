#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct gt
{
    std::vector<std::string> input;
    std::vector<int> rank;
    std::vector<long long> encoded;
};

class LinearModel
{
public:
    void init(std::vector<std::string> dt);
    int forward(std::string k);
    void adjust(double dl_dk, double dl_db, double lr);
    gt data;
    int error;
private:
    int k;
    int b;
    long long max_encoded;
    long long min_encoded;
};

#endif
