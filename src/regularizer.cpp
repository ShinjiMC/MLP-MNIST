#include "regularizer.hpp"
#include "layer.hpp"

double L2Regularizer::compute_penalty(const std::vector<Layer> &layers) const
{
    double penalty = 0.0;
    for (const auto &layer : layers)
        layer.compute_penalty(penalty);
    return 0.5 * lambda * penalty;
}

void L2Regularizer::apply(std::vector<double> &weights, std::vector<double> &grad_weights) const
{
    for (size_t i = 0; i < weights.size(); ++i)
        grad_weights[i] += lambda * weights[i];
}