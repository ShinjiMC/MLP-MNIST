#include "layer.hpp"

Layer::Layer(int n_neurons, int n_inputs_per_neuron,
             std::function<float(float)> act,
             std::function<float(float)> act_deriv)
    : activation(act), activation_derivative(act_deriv)
{
    for (int i = 0; i < n_neurons; ++i)
        neurons.emplace_back(n_inputs_per_neuron, activation);
}

void Layer::load_layer(int n_neurons, int n_inputs_per_neuron,
                       std::function<float(float)> act,
                       std::function<float(float)> act_deriv,
                       const std::vector<std::vector<float>> &all_weights = {},
                       const std::vector<float> &all_biases = {})
{
    activation = act;
    activation_derivative = act_deriv;
    neurons.clear();
    neurons.reserve(n_neurons);
    for (int i = 0; i < n_neurons; ++i)
    {
        neurons.emplace_back(n_inputs_per_neuron, act);
        if (!all_weights.empty() && i < (int)all_weights.size())
            neurons.back().set_weights(all_weights[i]);
        if (!all_biases.empty() && i < (int)all_biases.size())
            neurons.back().set_sesgo(all_biases[i]);
    }
    last_input.clear();
    last_z.clear();
}

std::vector<float> Layer::forward(const std::vector<float> &inputs)
{
    last_input = inputs;
    last_z.clear();
    std::vector<float> outputs;
    for (auto &n : neurons)
    {
        float z = 0.0f;
        auto weights = n.get_weights();
        for (size_t i = 0; i < weights.size(); ++i)
            z += weights[i] * inputs[i];
        z += n.get_sesgo();
        last_z.push_back(z);
        outputs.push_back(activation(z));
    }
    return outputs;
}

std::vector<float> Layer::backward(const std::vector<float> &deltas_next,
                                   const std::vector<std::vector<float>> &weights_next)
{
    std::vector<float> deltas(neurons.size(), 0.0f);
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < deltas_next.size(); ++j)
            sum += weights_next[j][i] * deltas_next[j];
        deltas[i] = sum * activation_derivative(last_z[i]);
    }
    return deltas;
}

void Layer::update_weights(float lr, const std::vector<float> &deltas)
{
    for (size_t i = 0; i < neurons.size(); ++i)
        neurons[i].update_weights(last_input, deltas[i], lr);
}

const std::vector<float> &Layer::get_last_input() const
{
    return last_input;
}

const std::vector<float> &Layer::get_last_z() const
{
    return last_z;
}

const std::vector<Neuron> &Layer::get_neurons() const
{
    return neurons;
}
