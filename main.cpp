#include "conv2d.hpp"
#include "pooling.hpp"
#include "flatten.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

int main()
{
    const int height = 3, width = 3, channels = 3;
    std::vector<std::vector<std::vector<double>>> input(channels, std::vector<std::vector<double>>(height, std::vector<double>(width)));
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                input[c][i][j] = c * 10 + i * 3 + j + 1;

    std::cout << "Imagen de entrada (3 canales, 3x3):\n";
    for (int c = 0; c < channels; ++c)
    {
        std::cout << "Canal " << c << ":\n";
        for (const auto &row : input[c])
        {
            for (double val : row)
                std::cout << val << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    Conv2D conv(3, 3, 2, 2, 1, 0, ActivationType::RELU); // in_channels=3, out_channels=2, kernel=2x2, stride=1, padding=0, activation=ReLU
    auto conv_output = conv.forward(input);
    std::cout << "Salida de la convolución (3 canales):\n";
    for (int c = 0; c < conv_output.size(); ++c)
    {
        std::cout << "Canal " << c << ":\n";
        for (const auto &row : conv_output[c])
        {
            for (double val : row)
                std::cout << val << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    Pooling pool(2, 2, 1, 0, PoolingType::MAX); // kernel=2x2, stride=1, padding=0, tipo=MAX
    auto pooled_output = pool.forward(conv_output);

    std::cout << "Salida del pooling (MAX):\n";
    for (int c = 0; c < pooled_output.size(); ++c)
    {
        std::cout << "Canal " << c << ":\n";
        for (const auto &row : pooled_output[c])
        {
            for (double val : row)
                std::cout << val << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    Flatten flatten;
    auto flat_output = flatten.forward(pooled_output);

    std::cout << "Salida del flatten (vector 1D):\n";
    for (double val : flat_output)
        std::cout << val << " ";
    std::cout << "\n";

    return 0;
}
