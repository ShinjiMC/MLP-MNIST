#pragma once
#include <string>
#include <vector>

class Dataset
{
private:
    std::vector<std::vector<float>> X;
    std::vector<float> y;

public:
    Dataset(const std::string &filename);
    const std::vector<std::vector<float>> &get_X() const;
    const std::vector<float> &get_y() const;
};