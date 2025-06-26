#pragma once
#include <cmath>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <algorithm>

enum ActivationType
{
    SIGMOID,
    RELU,
    TANH
};

inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

inline double relu(double x)
{
    return x > 0 ? x : 0;
}

inline double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

inline double tanh_fn(double x)
{
    return std::tanh(x);
}

inline double tanh_derivative(double x)
{
    double t = std::tanh(x);
    return 1.0 - t * t;
}