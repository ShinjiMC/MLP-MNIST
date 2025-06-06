#pragma once
#include <random>
#include "neuron.hpp"
#include "activation.hpp"

class Layer
{
private:
    int input_size;
    int output_size;
    std::vector<Neuron> neurons;
    ActivationType activation;

public:
    Layer(int in_size, int out_size, ActivationType act)
        : input_size(in_size), output_size(out_size), activation(act)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<> dis(-limit, limit);

        for (int i = 0; i < output_size; ++i)
            neurons.emplace_back(input_size, gen, dis);
    }
    Layer(int in_size, int out_size, ActivationType act, bool true_random)
        : input_size(in_size), output_size(out_size), activation(act)
    {
        neurons.resize(output_size, Neuron());
    }
    void linear_forward(const std::vector<double> &input, std::vector<double> &output);

    // gets
    int get_input_size() { return input_size; }
    int get_output_size() { return output_size; }
    const int get_inputss() const { return input_size; }
    const int get_outputss() const { return output_size; }
    std::vector<Neuron> &get_neurons() { return neurons; }
    const std::vector<Neuron> &get_neuronss() const { return neurons; }
    ActivationType get_activation() const { return activation; }
    const int get_neurons_size() const { return neurons.size(); }
    void save(std::ostream &out, const int i) const;
    void load(std::istream &in);
};