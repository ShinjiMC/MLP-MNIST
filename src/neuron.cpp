#include "neuron.hpp"
#include <random>
#include <iostream>

Neuron::Neuron(int n_inputs, std::function<float(float)> activation_func)
{
    if (n_inputs <= 0)
        throw std::invalid_argument("Number of inputs must be positive");

    srand(47);
    weights.resize(n_inputs, 0);
    for (int i = 0; i < n_inputs; i++)
        weights[i] = static_cast<float>(rand()) / RAND_MAX;
    sesgo = 0.0f;
    activation = activation_func ? activation_func : [](float x)
    { return x >= 0 ? 1.0f : 0.0f; };
}

void Neuron::set_weights(const std::vector<float> &new_weights)
{
    if (new_weights.size() != weights.size())
        throw std::invalid_argument("Size of weights /= n inputs");
    weights = new_weights;
}

float Neuron::get_weight(int id) const
{
    if (id < 0 || id >= weights.size())
        throw std::out_of_range("Id out of range");
    return weights[id];
}

std::vector<float> Neuron::get_weights() const
{
    return weights;
}

void Neuron::set_sesgo(float new_sesgo)
{
    sesgo = new_sesgo;
}

float Neuron::get_sesgo() const
{
    return sesgo;
}

void Neuron::update_weights(const std::vector<float> &inputs, float err, float lr)
{
    if (inputs.size() != weights.size())
        throw std::invalid_argument("Size of inputs /= n inputs");
    for (size_t i = 0; i < weights.size(); i++)
        weights[i] += lr * err * inputs[i];
    sesgo += lr * err;
}

float Neuron::forward(const std::vector<float> &inputs)
{
    if (inputs.size() != weights.size())
    {
        std::cerr << "Error: inputs.size() = " << inputs.size()
                  << " pero la neurona esperaba " << weights.size() << std::endl;
        throw std::invalid_argument("Size of inputs /= n inputs");
    }
    float a = sesgo;
    for (size_t i = 0; i < weights.size(); i++)
        a += weights[i] * inputs[i];
    return activation(a);
}