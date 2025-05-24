#include "dataset.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
Dataset::Dataset(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cant open file: " + filename);

    int n_inputs, n_options;
    file >> n_inputs >> n_options;
    if (n_inputs < 1 || n_options < 1)
        throw std::runtime_error("Invalid number of inputs or options in file: " + filename);
    // std::cout << "n_inputs: " << n_inputs << ", n_options: " << n_options << std::endl;
    X.clear();
    y.clear();

    for (size_t i = 0; i < n_options; i++)
    {
        float output;
        std::vector<float> row(n_inputs);
        for (size_t j = 0; j < n_inputs; j++)
            file >> row[j];
        file >> output;
        X.push_back(row);
        y.push_back(output);
    }
    file.close();
}

const std::vector<std::vector<float>> &Dataset::get_X() const
{
    return X;
}

const std::vector<float> &Dataset::get_y() const
{
    return y;
}