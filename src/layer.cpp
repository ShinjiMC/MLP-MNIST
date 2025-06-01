#include "layer.hpp"
void Layer::linear_forward(const std::vector<double> &input, std::vector<double> &output)
{
    for (int i = 0; i < this->output_size; ++i)
    {
        output[i] = this->neurons[i].get_biass();
        for (int j = 0; j < this->input_size; ++j)
        {
            output[i] += input[j] * this->neurons[i].get_weightss()[j];
        }
    }
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
    else if (this->activation == SOFTMAX)
    {
        softmax(output, output);
    }
    else if (this->activation == TANH)
    {
        for (double &val : output)
            val = tanh(val);
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
