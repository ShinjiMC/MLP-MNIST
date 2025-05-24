#include <iostream>
#include "mlp.hpp"

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cout << "Use: ./main_test x1 x2 mlp_saved.txt" << std::endl;
        return 1;
    }
    float x1 = std::stof(argv[1]);
    float x2 = std::stof(argv[2]);
    std::string filename = argv[3];
    MLP mlp;
    if (!mlp.load_from_file(filename))
    {
        std::cerr << "Error: Could not load MLP from file: " << filename << std::endl;
        return 1;
    }
    std::cout << "MLP loaded successfully from " << filename << std::endl;
    mlp.print_weights(0, 0.0f);
    std::vector<std::vector<float>> X = {{x1, x2}};
    std::vector<float> pred = mlp.predict(X[0]);
    std::cout << "Predictions:\n";
    std::cout << "(";
    for (float val : X[0])
        std::cout << val << " ";
    std::cout << ") => " << pred[0] << " ≈ " << (pred[0] > 0.5f ? 1 : 0) << "\n";
    return 0;
}