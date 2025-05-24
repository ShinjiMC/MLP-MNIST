#pragma once
#include <cmath>

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoid_derivative(float x)
{
    float s = sigmoid(x);
    return s * (1.0f - s);
}

inline float relu(float x)
{
    return x > 0 ? x : 0;
}

inline float relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

inline float tanh_fn(float x)
{
    return std::tanh(x);
}

inline float tanh_derivative(float x)
{
    float t = std::tanh(x);
    return 1.0f - t * t;
}

inline std::unordered_map<std::string, std::function<float(float)>> activation_map = {
    {"sigmoid", sigmoid},
    {"relu", relu},
    {"tanh", tanh_fn}};

inline std::unordered_map<std::string, std::function<float(float)>> derivative_map = {
    {"sigmoid", sigmoid_derivative},
    {"relu", relu_derivative},
    {"tanh", tanh_derivative}};

inline std::string get_activation_name(const std::function<float(float)> &func)
{
    auto ptr = func.target<float (*)(float)>();
    if (ptr && *ptr == sigmoid)
        return "sigmoid";
    else if (ptr && *ptr == relu)
        return "relu";
    else if (ptr && *ptr == tanh_fn)
        return "tanh";
    else
        return "unknown";
}

inline std::string get_derivative_name(const std::function<float(float)> &func)
{
    auto ptr = func.target<float (*)(float)>();
    if (ptr && *ptr == sigmoid_derivative)
        return "sigmoid";
    else if (ptr && *ptr == relu_derivative)
        return "relu";
    else if (ptr && *ptr == tanh_derivative)
        return "tanh";
    else
        return "unknown";
}
