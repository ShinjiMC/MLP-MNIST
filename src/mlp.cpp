#include "mlp.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

MLP::MLP(const std::vector<int> &sizes,
         const std::vector<std::function<float(float)>> &activations,
         const std::vector<std::function<float(float)>> &derivatives,
         float lr)
    : learning_rate(lr), activations(activations), derivatives(derivatives),
      input_size(sizes[0])
{
    if (sizes.size() < 2 || activations.size() != sizes.size() - 1)
        throw std::invalid_argument("Invalid layer or activation sizes");
    for (size_t i = 1; i < sizes.size(); ++i)
        layers.emplace_back(Layer(sizes[i], sizes[i - 1], activations[i - 1], derivatives[i - 1]));
}

std::vector<float> MLP::predict(const std::vector<float> &input)
{
    std::vector<float> output = input;
    for (auto &layer : layers)
        output = layer.forward(output);
    return output;
}

float MLP::mse(const std::vector<float> &pred, const std::vector<float> &target)
{
    float sum = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i)
        sum += (pred[i] - target[i]) * (pred[i] - target[i]);
    return sum / pred.size();
}

void MLP::backpropagate(const std::vector<float> &output,
                        const std::vector<float> &target,
                        std::vector<std::vector<float>> &all_deltas)
{
    // Salida
    std::vector<float> delta_output(output.size(), 0.0f);
    const auto &z_out = layers.back().get_last_z();
    auto act_deriv = layers.back().get_activation_derivative();
    for (size_t j = 0; j < output.size(); ++j)
        delta_output[j] = (target[j] - output[j]) * act_deriv(z_out[j]);

    all_deltas.back() = delta_output;

    // Propagación inversa
    for (int l = layers.size() - 2; l >= 0; --l)
    {
        std::vector<std::vector<float>> next_weights;
        for (const auto &n : layers[l + 1].get_neurons())
            next_weights.push_back(n.get_weights());

        all_deltas[l] = layers[l].backward(all_deltas[l + 1], next_weights);
    }
}

std::pair<float, float> MLP::train_epoch(const std::vector<std::vector<float>> &X,
                                         const std::vector<std::vector<float>> &Y)
{
    float error_total = 0.0f;
    float correct = 0.0f;
    for (size_t i = 0; i < X.size(); ++i)
    {
        std::vector<float> output = predict(X[i]);

        // Accuracy
        bool correct_prediction = false;
        bool is_binary = output.size() == 1;
        if (is_binary)
        {
            bool predicted = output[0] >= 0.5f;
            bool expected = Y[i][0] >= 0.5f;
            correct_prediction = predicted == expected;
        }
        else
        {
            int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            int expected = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
            correct_prediction = predicted == expected;
        }
        if (correct_prediction)
            ++correct;

        // Backpropagation y update weights
        std::vector<std::vector<float>> all_deltas(layers.size());
        backpropagate(output, Y[i], all_deltas);
        for (size_t l = 0; l < layers.size(); ++l)
            layers[l].update_weights(learning_rate, all_deltas[l]);
        error_total += mse(output, Y[i]);
    }
    float mse_avg = error_total / X.size();
    float acc = correct / X.size();
    return {mse_avg, acc};
}

void MLP::train(const std::vector<std::vector<float>> &X,
                const std::vector<std::vector<float>> &Y,
                float min_error, bool print,
                const std::string &dataset_filename)
{
    std::string base_name = std::filesystem::path(dataset_filename).stem().string();
    std::filesystem::path output_dir = std::filesystem::path("output") / base_name;
    std::filesystem::create_directories(output_dir);
    std::ofstream log_file(output_dir / "log.txt");
    if (!log_file)
    {
        std::cerr << "Can't open '" << output_dir / "log.txt" << "' for writing.\n";
        return;
    }
    int e = 0;
    float last_mse = 0.0f;
    while (true)
    {
        auto [mse_avg, acc] = train_epoch(X, Y);
        log_epoch(log_file, e, mse_avg, acc);
        if (print)
            print_weights(e, mse_avg);
        if (mse_avg < min_error)
        {
            std::cout << "Training stopped at epoch " << e
                      << " with MSE: " << mse_avg << "\n";
            break;
        }
        if (e == 4000000 && mse_avg == last_mse)
        {
            std::cout << "Training stopped at epoch " << e
                      << " with MSE: " << mse_avg << "\n";
            break;
        }
        e++;
        last_mse = mse_avg;
    }
    log_file.close();
    save_final_weights((output_dir / "final.txt").string());
}

void MLP::log_epoch(std::ofstream &log_file, int epoch, float mse_avg, float acc)
{
    log_file << "Epoch " << epoch << " - MSE: " << mse_avg << " - ACC: " << acc << "\n";
}

void MLP::save_final_weights(const std::string &path)
{
    std::ofstream final_file(path);
    if (!final_file)
    {
        std::cerr << "Cant open '" << path << "' for writing.\n";
        return;
    }
    final_file << input_size << " ";
    for (size_t i = 0; i < layers.size(); ++i)
        final_file << layers[i].get_neurons().size() << " ";
    final_file << learning_rate << "\n";

    for (size_t i = 0; i < activations.size(); ++i)
        final_file << get_activation_name(activations[i]) << " ";
    final_file << "\n";

    for (size_t i = 0; i < derivatives.size(); ++i)
        final_file << get_derivative_name(derivatives[i]) << " ";
    final_file << "\n";
    for (size_t l = 0; l < layers.size(); ++l)
        for (size_t n = 0; n < layers[l].get_neurons().size(); ++n)
        {
            const Neuron &neuron = layers[l].get_neurons()[n];
            final_file << l + 1 << " " << n + 1 << " ";
            for (auto weight : neuron.get_weights())
                final_file << weight << " ";
            final_file << " " << neuron.get_sesgo() << "\n";
        }
    final_file.close();
}

void MLP::print_weights(int epoch, float mse_avg)
{
    std::cout << "Epoch " << epoch << " - MSE: " << mse_avg << "\n";
    std::cout << "Layer | Neuron | Weights | Bias\n";
    for (size_t l = 0; l < layers.size(); ++l)
    {
        for (size_t n = 0; n < layers[l].get_neurons().size(); ++n)
        {
            const Neuron &neuron = layers[l].get_neurons()[n];
            std::cout << l + 1 << " | " << n + 1 << " | ";
            for (auto weight : neuron.get_weights())
                std::cout << weight << " ";
            std::cout << "| " << neuron.get_sesgo() << "\n";
        }
    }
}

bool MLP::load_from_file(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // Leer la primera línea completa
    std::string line;
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read first line from file.\n";
        return false;
    }

    std::istringstream iss(line);
    float possible_float;
    std::vector<int> layer_sizes;
    bool found_lr = false;
    while (iss >> possible_float)
    {
        int as_int = static_cast<int>(possible_float);
        if (possible_float == as_int)
        {
            layer_sizes.push_back(as_int);
        }
        else
        {
            this->learning_rate = possible_float;
            found_lr = true;
            break;
        }
    }

    if (!found_lr)
    {
        if (!(iss >> this->learning_rate))
        {
            std::cerr << "Error: Could not read learning rate.\n";
            return false;
        }
    }

    if (layer_sizes.empty())
    {
        std::cerr << "Error: No layer sizes found in file.\n";
        return false;
    }

    std::vector<std::function<float(float)>> activs;
    for (int i = 0; i < (int)layer_sizes.size() - 1; ++i)
    {
        std::string act_name;
        if (!(file >> act_name))
        {
            std::cerr << "Error: Could not read activation function name.\n";
            return false;
        }
        auto it = activation_map.find(act_name);
        if (it == activation_map.end())
        {
            std::cerr << "Error: Unknown activation function: " << act_name << std::endl;
            return false;
        }
        activs.push_back(it->second);
    }

    std::vector<std::function<float(float)>> derivs;
    for (int i = 0; i < (int)layer_sizes.size() - 1; ++i)
    {
        std::string deriv_name;
        if (!(file >> deriv_name))
        {
            std::cerr << "Error: Could not read derivative function name.\n";
            return false;
        }
        auto it = derivative_map.find(deriv_name);
        if (it == derivative_map.end())
        {
            std::cerr << "Error: Unknown derivative function: " << deriv_name << std::endl;
            return false;
        }
        derivs.push_back(it->second);
    }

    std::vector<std::vector<std::vector<float>>> all_weights(layer_sizes.size() - 1);
    std::vector<std::vector<float>> all_biases(layer_sizes.size() - 1);
    for (size_t i = 0; i < all_weights.size(); ++i)
    {
        all_weights[i].resize(layer_sizes[i + 1]);
        all_biases[i].resize(layer_sizes[i + 1]);
    }

    while (true)
    {
        int layer_idx, neuron_idx;
        if (!(file >> layer_idx >> neuron_idx))
            break;
        layer_idx -= 1;
        neuron_idx -= 1;
        if (layer_idx < 0 || layer_idx >= (int)all_weights.size())
        {
            std::cerr << "Error: Invalid layer index " << layer_idx + 1 << std::endl;
            return false;
        }
        if (neuron_idx < 0 || neuron_idx >= (int)all_weights[layer_idx].size())
        {
            std::cerr << "Error: Invalid neuron index " << neuron_idx + 1 << " in layer " << layer_idx + 1 << std::endl;
            return false;
        }
        int n_inputs_per_neuron = layer_sizes[layer_idx];
        std::vector<float> weights(n_inputs_per_neuron);
        for (int w_i = 0; w_i < n_inputs_per_neuron; ++w_i)
        {
            if (!(file >> weights[w_i]))
            {
                std::cerr << "Error: Could not read weight #" << w_i << " for neuron " << neuron_idx + 1 << " in layer " << layer_idx + 1 << std::endl;
                return false;
            }
        }
        float bias;
        if (!(file >> bias))
        {
            std::cerr << "Error: Could not read bias for neuron " << neuron_idx + 1 << " in layer " << layer_idx + 1 << std::endl;
            return false;
        }
        all_weights[layer_idx][neuron_idx] = std::move(weights);
        all_biases[layer_idx][neuron_idx] = bias;
    }

    file.close();
    layers.clear();
    for (size_t i = 1; i < layer_sizes.size(); ++i)
    {
        int n_neurons = layer_sizes[i];
        int n_inputs_per_neuron = layer_sizes[i - 1];
        layers.emplace_back();
        layers.back().load_layer(n_neurons, n_inputs_per_neuron, activs[i - 1],
                                 derivs[i - 1], all_weights[i - 1],
                                 all_biases[i - 1]);
    }

    return true;
}
