#include "mlp.hpp"

Mlp::Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
         double lr, std::vector<ActivationType> activation_types)
    : n_inputs(n_inputs), n_outputs(n_outputs), learning_rate(lr)
{
    if (layer_sizes.size() != activation_types.size())
    {
        throw std::invalid_argument("layer_sizes and activation_types must have the same length.");
    }

    int prev_size = n_inputs;

    for (size_t i = 0; i < layer_sizes.size(); ++i)
    {
        std::cout << "Creating layer " << i + 1
                  << " with " << layer_sizes[i]
                  << " neurons and activation " << activation_types[i] << ".\n";
        layers.emplace_back(prev_size, layer_sizes[i], activation_types[i]);
        prev_size = layer_sizes[i];
    }
}

void Mlp::forward(const std::vector<double> &input, std::vector<std::vector<double>> &activations)
{
    activations.clear();
    activations.push_back(input);

    for (auto &layer : layers)
    {
        std::vector<double> output(layer.get_output_size());
        layer.linear_forward(activations.back(), output);
        activations.push_back(output);
    }
}
void Mlp::backward(const std::vector<double> &input,
                   const std::vector<std::vector<double>> &activations,
                   const std::vector<double> &expected)
{
    std::vector<std::vector<double>> deltas(layers.size());
    for (int l = (int)layers.size() - 1; l >= 0; --l)
    {
        int n_neurons = layers[l].get_output_size();
        deltas[l].resize(n_neurons);
        if (layers[l].get_activation() == SOFTMAX)
            for (int i = 0; i < n_neurons; ++i)
                deltas[l][i] = activations[l + 1][i] - expected[i];
        else
        {
            for (int i = 0; i < n_neurons; ++i)
            {
                double error = 0.0;
                for (int j = 0; j < layers[l + 1].get_output_size(); ++j)
                    error += deltas[l + 1][j] * layers[l + 1].get_neurons()[j].get_weights()[i];
                if (layers[l].get_activation() == RELU)
                    deltas[l][i] = error * relu_derivative(activations[l + 1][i]);
                else if (layers[l].get_activation() == SIGMOID)
                    deltas[l][i] = error * sigmoid_derivative(activations[l + 1][i]);
                else // TANH
                    deltas[l][i] = error * tanh_derivative(activations[l + 1][i]);
            }
        }
    }

    // Actualización de pesos y biases
    for (size_t l = 0; l < layers.size(); ++l)
    {
        for (int i = 0; i < layers[l].get_output_size(); ++i)
        {
            for (int j = 0; j < layers[l].get_input_size(); ++j)
                layers[l].get_neurons()[i].get_weights()[j] -= learning_rate * deltas[l][i] * activations[l][j];
            layers[l].get_neurons()[i].get_bias() -= learning_rate * deltas[l][i];
        }
    }
}

void Mlp::train(std::vector<std::vector<double>> &images, std::vector<int> &labels,
                double &average_loss, double &train_accuracy)
{
    size_t n = images.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    std::vector<double> target(n_outputs);
    std::vector<std::vector<double>> activations;
    double total_loss = 0.0;
    int correct = 0;

    for (size_t k = 0; k < n; ++k)
    {
        size_t i = indices[k];
        const auto &input = images[i];
        int label = labels[i];
        one_hot_encode(label, target);
        forward(input, activations);

        total_loss += cross_entropy_loss(activations.back(), target);
        backward(input, activations, target);

        int pred = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == label)
            ++correct;
    }
    average_loss = total_loss / n;
    train_accuracy = 100.0 * correct / n;
}

void Mlp::test(const std::vector<std::vector<double>> &images, const std::vector<int> &labels, double &test_accuracy)
{
    int correct = 0;
    std::vector<std::vector<double>> activations;

    for (size_t i = 0; i < images.size(); ++i)
    {
        forward(images[i], activations);
        int pred = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == labels[i])
            ++correct;
    }

    test_accuracy = 100.0 * correct / images.size();
}

void Mlp::train_test(std::vector<std::vector<double>> &train_images, std::vector<int> &train_labels,
                     const std::vector<std::vector<double>> &test_images, const std::vector<int> &test_labels,
                     bool Test, const std::string &dataset_filename, int epochs)
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
    int epoch = 0;
    double average_loss = 0, train_accuracy = 0, test_accuracy = 0;
    while (true)
    {
        // --- Entrenamiento ---
        clock_t train_start = clock();
        train(train_images, train_labels, average_loss, train_accuracy);
        clock_t train_end = clock();
        double train_time = double(train_end - train_start) / CLOCKS_PER_SEC;

        // --- Evaluación en test (accuracy) ---
        double test_time = 0.0;
        if (Test)
        {
            clock_t test_start = clock();
            test(test_images, test_labels, test_accuracy);
            clock_t test_end = clock();
            test_time = double(test_end - test_start) / CLOCKS_PER_SEC;
        }

        std::ostringstream log_line;
        log_line << "Epoch " << (epoch + 1)
                 << ", Train Loss: " << average_loss
                 << ", Train Acc: " << train_accuracy << "%"
                 << ", Train Time: " << train_time << "s";

        if (Test)
            log_line << ", Test Acc: " << test_accuracy << "%"
                     << ", Test Time: " << test_time << "s";

        std::cout << log_line.str() << std::endl;
        log_file << log_line.str() << std::endl;

        if (average_loss < 0.001 || test_accuracy > 98.0 || epoch >= epochs)
        {
            std::cout << "Stopping training: early stopping criteria met.\n";
            break;
        }
        epoch++;
    }
}

void Mlp::save_data(const std::string &filename) const
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Error: no se pudo abrir " << filename << " para guardar.\n";
        return;
    }

    // Cabecera
    out << n_inputs << " ";
    for (auto a : layers)
        out << a.get_neurons_size() << " ";
    out << "\n"
        << learning_rate << "\n";

    // Tipos de activación por capa
    for (const auto &layer : layers)
        out << to_string(layer.get_activation()) << " ";
    out << "\n";

    // Capas y neuronas
    for (size_t i = 0; i < layers.size(); ++i)
        layers[i].save(out, i);
    out.close();
}

bool Mlp::load_data(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in)
    {
        std::cerr << "Error: no se pudo abrir " << filename << " para leer.\n";
        return false;
    }

    std::cout << "Cargando MLP desde " << filename << "...\n";
    // Leer arquitectura
    std::vector<int> layer_sizes;
    std::string line;
    std::getline(in, line);
    std::istringstream arch_stream(line);
    arch_stream >> n_inputs;
    int size;
    while (arch_stream >> size)
        layer_sizes.push_back(size);
    n_outputs = layer_sizes.back();
    layer_sizes.pop_back();
    in >> learning_rate;

    // Leer funciones de activación
    std::vector<ActivationType> activations(layer_sizes.size() + 1);
    for (size_t i = 0; i < activations.size(); ++i)
    {
        std::string act;
        in >> act;
        activations[i] = from_string(act);
    }

    // Construir capas y cargar datos
    layers.clear();
    int prev_size = n_inputs;
    for (size_t i = 0; i < activations.size(); ++i)
    {
        int curr_size = (i < layer_sizes.size()) ? layer_sizes[i] : n_outputs;
        Layer layer(prev_size, curr_size, activations[i], true);
        layer.load(in);
        layers.push_back(std::move(layer));
        prev_size = curr_size;
    }

    in.close();
    return true;
}

void Mlp::test_info(const std::vector<std::vector<double>> &X_test, const std::vector<int> &y_test)
{
    int correct = 0;
    std::vector<std::vector<double>> activations;

    for (size_t i = 0; i < X_test.size(); ++i)
    {
        forward(X_test[i], activations);
        int pred = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == y_test[i])
            ++correct;

        std::cout << "Image " << i << " - Output: " << pred << " - Correct: " << y_test[i] << "\n";
    }

    std::cout << "\nTotal: " << X_test.size() << "\n";
    std::cout << "Correct: " << correct << "\n";
    std::cout << "Accuracy: " << 100.0 * correct / X_test.size() << "%\n";
}
