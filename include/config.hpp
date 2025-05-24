#pragma once
#include <string>
#include <vector>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include "activation.hpp"

class Config
{
private:
    std::vector<int> layer_sizes;
    float learning_rate;
    std::vector<std::function<float(float)>> activations;
    std::vector<std::function<float(float)>> derivatives;

public:
    bool load_config(const std::string &filename, const int inputs);
    const std::vector<int> &get_layer_sizes() const { return layer_sizes; }
    float get_learning_rate() const { return learning_rate; }
    const std::vector<std::function<float(float)>> &get_activations() const { return activations; }
    const std::vector<std::function<float(float)>> &get_derivatives() const { return derivatives; }
    const void print_config();
};
