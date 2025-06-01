#include <fstream>
#include <cstdint>
#include "mlp.hpp"
#include "dataset.hpp"
#include "config.hpp"

constexpr int TRAIN_SAMPLES = 20000;
constexpr int TEST_SAMPLES = 5000;

int main()
{

    Dataset m;
    m.generate_mnist("./database/MNIST/", TRAIN_SAMPLES, TEST_SAMPLES);
    std::string dir = "./database/MNIST/";
    std::string train_file = dir + "train.txt";
    std::string test_file = dir + "test.txt";

    Dataset train(train_file);
    std::vector<std::vector<double>> X_train = train.get_X();
    std::vector<int> y_train = train.get_ys();

    Dataset test(test_file);
    std::vector<std::vector<double>> X_test = test.get_X();
    std::vector<int> y_test = test.get_ys();

    // Print de ejemplo uno de train y label
    std::cout << "Example 0 of training data:\n";
    for (int i = 0; i < 28 * 28; ++i)
    {
        std::cout << X_train[0][i] << " ";
        if ((i + 1) % 28 == 0)
            std::cout << "\n";
    }
    std::cout << "Label for example 0: " << y_train[0] << "\n";
    int n_inputs = X_train[0].size();

    Config cfg;
    cfg.load_config("config/mnist.txt", n_inputs);
    cfg.print_config();
    // Initialize neural network
    printf("Initializing neural network...\n");
    Mlp nn(n_inputs, cfg.get_layer_sizes(),
           cfg.get_layer_sizes().back(), cfg.get_learning_rate(),
           cfg.get_activations());

    // Train the neural network
    printf("Training neural network...\n");
    nn.train_test(X_train, y_train, X_test, y_test, true, "MNIST", 1);
    nn.save_data("./output/MNIST/mnist_mlp.dat");

    Mlp n2;
    if (!n2.load_data("./output/MNIST/mnist_mlp.dat"))
    {
        std::cerr << "Error loading MLP data.\n";
        return 1;
    }
    std::cout << "MLP loaded successfully.\n";
    n2.train_test(X_train, y_train, X_test, y_test, true, "MNIST2", 1);

    return 0;
}