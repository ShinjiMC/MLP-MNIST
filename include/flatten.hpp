#pragma once
#include <vector>

class Flatten
{
private:
    // Guarda la forma original para un posible reshape inverso
    int channels, height, width;

public:
    // Convierte un tensor [C][H][W] en un vector 1D
    std::vector<double> forward(const std::vector<std::vector<std::vector<double>>> &input);

    // Convierte un vector plano a la forma original [C][H][W]
    std::vector<std::vector<std::vector<double>>> reshape(const std::vector<double> &input_flat);
};
