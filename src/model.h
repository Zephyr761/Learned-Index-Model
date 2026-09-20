#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdint>

struct gt
{
    std::vector<std::string> input;
    std::vector<long long> encoded;
};

class LinearModel
{
public:
    void init(const std::vector<std::string>& dt);
    int forward_inference(std::string k);
    double forward_train(std::string k);
    void update(double dl_dk, double dl_db, double lr);
    gt data;
    int error;
private:
    double k;
    double b;
    long long max_encoded;
    long long min_encoded;
};

#endif
