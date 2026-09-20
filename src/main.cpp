#include "model.h"
#include <random>
#include <fstream>

std::vector<std::string> load_dataset(const std::string& path, int max)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if(!file) throw std::runtime_error("cannot open dataset: " + path);
    std::vector<std::string> keys;
    std::string line;
    int times = 0;
    while(std::getline(file, line) && times < max)
    {
        if(!line.empty() && line.back() == '\r') line.pop_back();
        keys.push_back(std::move(line));
        times++;
    }
    return keys;
}

void train(LinearModel& model, const std::vector<std::string>& dt, int steps, double lr, int bs)
{
    model.init(dt);
    double dl_dk = 0;
    double dl_db = 0;
    for(int s = 0; s < steps; ++s)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, dt.size() - 1);
        for(int i = 0; i < bs; ++i)
        {
            int index = dist(gen);
            auto rank = std::lower_bound(dt.begin(), dt.end(), model.data.input[index])
                        - dt.begin();
            double predicted = model.forward_train(model.data.input[index]);
            double loss = abs(predicted - rank) * 1.0;
            double sign = ((predicted - rank) > 0) - ((predicted - rank) < 0);
            dl_dk += sign * model.data.encoded[index];
            dl_db += sign;
            model.error = std::max(model.error, int(abs(predicted - rank)) + 1);
        }
        dl_dk /= bs;
        dl_db /= bs;
        model.update(dl_dk, dl_db, lr);
    }
}

int main()
{
    std::vector<std::string> key = load_dataset("/Users/zephyr/Projects/Learned Index/Learned-Index-Model/data/reddit_25865740_maxlen128", 1000);
    LinearModel lm;
    int step, lr, bs;
    std::cin >> step >> lr >> bs;
    train(lm, key, step, lr, bs);
}