#include "model.h"
#include <random>

void train(LinearModel& model, std::vector<std::string> dt, double lr, int bs)
{
    model.init(dt);
    double dl_dk = 0;
    double dl_db = 0;
    for(int i = 0; i < bs; ++i)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, dt.size());
        int index = dist(gen);
        int predicted = model.forward(model.data.input[index]);
        double loss = abs(predicted - model.data.rank[index]) * 1.0;
        dl_db += abs(predicted - model.data.rank[index]) /(predicted - model.data.rank[index]);
        dl_dk += dl_db * model.data.encoded[index];
    }
    dl_dk /= bs;
    dl_db /= bs;
    model.adjust(dl_dk, dl_db, lr);
}