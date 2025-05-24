#pragma once
#include <vector>
#include <functional>
#include <stdexcept>

class Neuron
{
private:
    std::vector<float> weights;
    float sesgo;
    std::function<float(float)> activation;

public:
    Neuron() = default;
    Neuron(int n_inputs, std::function<float(float)> activation_func);
    void set_weights(const std::vector<float> &new_weights);
    float get_weight(int id) const;
    std::vector<float> get_weights() const;
    void set_sesgo(float new_sesgo);
    float get_sesgo() const;

    // Update Values
    void update_weights(const std::vector<float> &inputs, float err, float lr);
    float forward(const std::vector<float> &inputs);
};