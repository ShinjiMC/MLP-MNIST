#pragma once
#include "layer.hpp"
#include "activation.hpp"

class MLP
{
private:
    std::vector<Layer> layers;
    float learning_rate;
    int input_size;
    std::vector<std::function<float(float)>> activations;
    std::vector<std::function<float(float)>> derivatives;

public:
    MLP() = default;
    MLP(const std::vector<int> &sizes,
        const std::vector<std::function<float(float)>> &activations,
        const std::vector<std::function<float(float)>> &derivatives,
        float lr = 0.1);

    std::vector<float> predict(const std::vector<float> &input);
    void train(const std::vector<std::vector<float>> &X,
               const std::vector<std::vector<float>> &Y,
               float min_error = 0.001f,
               bool print = false,
               const std::string &dataset_filename = "databaese.txt");
    float mse(const std::vector<float> &pred, const std::vector<float> &target);
    void backpropagate(const std::vector<float> &output,
                       const std::vector<float> &target,
                       std::vector<std::vector<float>> &all_deltas);
    std::pair<float, float> train_epoch(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &Y);

    void log_epoch(std::ofstream &log_file, int epoch, float mse_avg, float acc);
    void save_final_weights(const std::string &path);
    void print_weights(int epoch, float mse_avg);
    bool load_from_file(const std::string &filename);
};
