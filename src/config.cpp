#include "config.hpp"
bool Config::load_config(const std::string &filename, const int inputs)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // Layer Inputs
    if (inputs <= 0)
    {
        std::cerr << "Error: Invalid number of inputs: " << inputs << std::endl;
        return false;
    }
    this->layer_sizes.push_back(static_cast<int>(inputs));

    std::string line;
    std::istringstream iss;

    // Leer layer sizes (ocultas + salida)
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read layer sizes line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    int layer_size;
    while (iss >> layer_size)
        this->layer_sizes.push_back(layer_size);

    if (this->layer_sizes.empty())
    {
        std::cerr << "Error: No layer sizes found in file.\n";
        return false;
    }

    // Leer learning rate
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read learning rate line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    if (!(iss >> this->learning_rate))
    {
        std::cerr << "Error: Invalid learning rate.\n";
        return false;
    }

    // Leer funciones de activación
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read activation functions line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    int expected_count = static_cast<int>(this->layer_sizes.size()) - 1;
    for (int i = 0; i < expected_count; ++i)
    {
        std::string act_name;
        if (!(iss >> act_name))
        {
            std::cerr << "Error: Not enough activation functions.\n";
            return false;
        }

        auto it = activation_map.find(act_name);
        if (it == activation_map.end())
        {
            std::cerr << "Error: Unknown activation function: " << act_name << std::endl;
            return false;
        }

        this->activations.push_back(it->second);
    }

    // Leer funciones derivadas
    if (!std::getline(file, line))
    {
        std::cerr << "Error: Could not read derivative functions line.\n";
        return false;
    }
    iss.clear();
    iss.str(line);
    for (int i = 0; i < expected_count; ++i)
    {
        std::string deriv_name;
        if (!(iss >> deriv_name))
        {
            std::cerr << "Error: Not enough derivative functions.\n";
            return false;
        }

        auto it = derivative_map.find(deriv_name);
        if (it == derivative_map.end())
        {
            std::cerr << "Error: Unknown derivative function: " << deriv_name << std::endl;
            return false;
        }

        this->derivatives.push_back(it->second);
    }

    return true;
}

const void Config::print_config()
{
    std::cout << "==========================\n";
    std::cout << "CONFIGURATION\n";
    std::cout << "Layer Sizes: \n";
    for (const auto &size : layer_sizes)
        std::cout << "\t" << size << " neurons\n";
    std::cout << "Learning Rate: " << learning_rate << "\n";
    std::cout << "Activations: \n";
    for (const auto &act : activations)
        std::cout << "\t" << get_activation_name(act) << "\n";
    std::cout << "Derivatives: \n";
    for (const auto &deriv : derivatives)
        std::cout << "\t" << get_derivative_name(deriv) << "\n";
    std::cout << "==========================\n";
    std::cout << std::endl;
}