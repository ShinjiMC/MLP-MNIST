#pragma once
#include "neuron.hpp"

class Layer
{
private:
    std::vector<Neuron> neurons;
    std::function<float(float)> activation;
    std::function<float(float)> activation_derivative;
    std::vector<float> last_input;
    std::vector<float> last_z;

public:
    Layer() = default;
    Layer(int n_neurons, int n_inputs_per_neuron,
          std::function<float(float)> act,
          std::function<float(float)> act_deriv);
    void load_layer(int n_neurons, int n_inputs_per_neuron,
                    std::function<float(float)> act,
                    std::function<float(float)> act_deriv,
                    const std::vector<std::vector<float>> &all_weights,
                    const std::vector<float> &all_biases);
    std::vector<float> forward(const std::vector<float> &inputs);
    std::vector<float> backward(const std::vector<float> &deltas_next,
                                const std::vector<std::vector<float>> &weights_next);
    void update_weights(float lr, const std::vector<float> &deltas);
    const std::vector<float> &get_last_input() const;
    const std::vector<float> &get_last_z() const;
    const std::vector<Neuron> &get_neurons() const;
    std::function<float(float)> get_activation_derivative() const { return activation_derivative; }
};
