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
    int n_inputs;
    std::vector<int> layer_sizes;
    float learning_rate;
    std::vector<ActivationType> activations;

public:
    bool load_config(const std::string &filename, const int inputs);
    const std::vector<int> &get_layer_sizes() const { return layer_sizes; }
    float get_learning_rate() const { return learning_rate; }
    const std::vector<ActivationType> &get_activations() const { return activations; }
    const void print_config();
};
