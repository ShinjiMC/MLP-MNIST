#include "neuron.hpp"
#include <cmath>

void Neuron::save(std::ostream &out) const
{
    for (const auto &w : weights)
        out << w << " ";
    out << bias << "\n";
}

void Neuron::load(std::istream &in, int n_inputs)
{
    weights.resize(n_inputs);
    for (int i = 0; i < n_inputs; ++i)
        in >> weights[i];
    in >> bias;
}
