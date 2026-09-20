#include "model.h"

void LinearModel::init(const std::vector<std::string>& dt)
{
    this->data.input = dt;
    std::sort(this->data.input.begin(), this->data.input.end());
    this->data.encoded.resize(this->data.input.size());
    for(int i = 0; i < this->data.input.size(); ++i)
    {
        this->data.encoded[i] = 0;
        for(int j = 0; j < 8 && j < this->data.input[i].size(); ++j) this->data.encoded[i] = (this->data.encoded[i] << 8) | static_cast<unsigned char>(this->data.input[i][j]);
    }
    this->max_encoded = *std::max_element(this->data.encoded.begin(), this->data.encoded.end());
    this->min_encoded = *std::min_element(this->data.encoded.begin(), this->data.encoded.end());
    this->k = dt.size() - 1;
    this->b = 0;
    this->error = 0;
}

double LinearModel::forward_train(std::string k)
{
    std::uint64_t x = 0;
    for(int j = 0; j < 8 && j < k.size(); ++j) 
    {
        x = (x << 8) | static_cast<unsigned char>(k[j]);
    }
    double xn = (x - this->min_encoded)*1.0 / (this->max_encoded - this->min_encoded);
    return this->k * xn + b;
}

int LinearModel::forward_inference(std::string k)
{
    std::uint64_t x = 0;
    for(int j = 0; j < 8 && j < k.size(); ++j) 
    {
        x = (x << 8) | static_cast<unsigned char>(k[j]);
    }
    double xn = (x - this->min_encoded)*1.0 / (this->max_encoded - this->min_encoded);
    return int(this->k * xn + b);
}

void LinearModel::update(double dl_dk, double dl_db, double lr)
{
    this->k -= dl_dk * lr;
    this->b -= dl_db * lr;
}

