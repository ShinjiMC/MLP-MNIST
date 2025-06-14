#include "layer.hpp"
void Layer::linear_forward(const std::vector<double> &input,
                           std::vector<double> &output) const
{
    for (int i = 0; i < this->output_size; ++i)
    {
        output[i] = this->neurons[i].get_biass();
        for (int j = 0; j < this->input_size; ++j)
            output[i] += input[j] * this->neurons[i].get_weightss()[j];
    }
}

void Layer::apply_activation(std::vector<double> &output) const
{
    if (this->activation == SIGMOID)
    {
        for (double &val : output)
            val = sigmoid(val);
    }
    else if (this->activation == RELU)
    {
        for (double &val : output)
            val = relu(val);
    }
    else if (this->activation == TANH)
    {
        for (double &val : output)
            val = tanh(val);
    }
    else if (this->activation == SOFTMAX)
    {
        softmax(output, output);
    }
}

void Layer::save(std::ostream &out, const int i) const
{
    for (size_t j = 0; j < neurons.size(); ++j)
    {
        out << i + 1 << " " << j + 1 << " ";
        neurons[j].save(out);
    }
}

void Layer::load(std::istream &in)
{
    neurons.resize(output_size);
    for (int j = 0; j < output_size; ++j)
    {
        int layer_idx, neuron_idx;
        in >> layer_idx >> neuron_idx;
        neurons[j].load(in, input_size);
    }
}

void Layer::apply_update(std::shared_ptr<Optimizer> optimizer, const std::vector<double> &delta,
                         const std::vector<double> &input,
                         double learning_rate, int layer_index)
{
    for (int i = 0; i < this->output_size; ++i)
        neurons[i].update(optimizer, learning_rate, input.data(), delta[i], this->input_size, i, layer_index);
}