#pragma once
#include <vector>
#include <random>

class Conv2D
{
private:
    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride, padding;

    // Filtros: [out_channels][in_channels][kernel_h][kernel_w]
    std::vector<std::vector<std::vector<std::vector<double>>>> filters;
    std::vector<double> biases;

    // gradientes calculados y entradas para backward
    std::vector<std::vector<std::vector<std::vector<double>>>> d_filters;
    std::vector<double> d_biases;
    std::vector<std::vector<std::vector<double>>> last_input;

    std::vector<std::vector<std::vector<double>>> pad_input(
        const std::vector<std::vector<std::vector<double>>> &input);
    void initialize_filters();

public:
    Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int stride = 1, int padding = 0);
    // Entrada: tensor XD [channels][height][width]
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>> &input);
    // retropropagación
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<std::vector<double>>> &grad_output);
};
