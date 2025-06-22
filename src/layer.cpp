#include "layer.hpp"
void Layer::linear_forward(const std::vector<double> &input,
                           std::vector<double> &output) const
{
    output.resize(output_size);
#pragma omp parallel for
    for (int i = 0; i < output_size; ++i)
    {
        double sum = biases[i];
        for (int j = 0; j < input_size; ++j)
            sum += weights[i][j] * input[j];
        output[i] = sum;
    }
}
void Layer::apply_activation(std::vector<double> &output) const
{
    if (this->activation == SIGMOID)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)output.size(); ++i)
            output[i] = sigmoid(output[i]);
    }
    else if (this->activation == RELU)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)output.size(); ++i)
            output[i] = relu(output[i]);
    }
    else if (this->activation == TANH)
    {
#pragma omp parallel for
        for (int i = 0; i < (int)output.size(); ++i)
            output[i] = tanh(output[i]);
    }
    else if (this->activation == SOFTMAX)
        softmax(output, output);
}

void Layer::save(std::ostream &out, const int i) const
{
    for (size_t j = 0; j < neurons.size(); ++j)
    {
        out << i + 1 << " " << j + 1 << " ";
        for (int k = 0; k < input_size; ++k)
            out << weights[j][k] << " ";
        out << biases[j] << "\n";
    }
}

void Layer::load(std::istream &in)
{
    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size);
    // neurons.resize(output_size);
    for (int j = 0; j < output_size; ++j)
    {
        int layer_idx, neuron_idx;
        in >> layer_idx >> neuron_idx;
        // neurons[j].load(in, input_size);
        for (int k = 0; k < input_size; ++k)
            in >> weights[j][k];
        in >> biases[j];
    }
}

void Layer::apply_update(std::shared_ptr<Optimizer> optimizer, const std::vector<double> &delta,
                         const std::vector<double> &input,
                         double learning_rate, int layer_index)
{
    // for (int i = 0; i < this->output_size; ++i)
    //     neurons[i].update(optimizer, learning_rate, input.data(), delta[i], this->input_size, i, layer_index);
    for (int i = 0; i < output_size; ++i)
    {
        int global_id = layer_index * 100000 + i;
        optimizer->update(learning_rate, weights[i], biases[i],
                          input.data(), delta[i], input_size, global_id);
    }
}

void Layer::compute_penalty(double &penalty) const
{
    // for (const auto &neuron : neurons)
    //     neuron.compute_penalty(penalty);
    for (const auto &row : weights)
        for (double w : row)
            penalty += w * w;
}