#include "pooling.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

Pooling::Pooling(int kernel_h, int kernel_w, int stride, int padding, PoolingType type)
    : kernel_h(kernel_h), kernel_w(kernel_w), stride(stride), padding(padding), type(type) {}

double Pooling::pool_region(const std::vector<std::vector<double>> &region) const
{
    std::vector<double> flat;
    for (const auto &row : region)
        flat.insert(flat.end(), row.begin(), row.end());

    switch (type)
    {
    case PoolingType::MAX:
        return *std::max_element(flat.begin(), flat.end());
    case PoolingType::MIN:
        return *std::min_element(flat.begin(), flat.end());
    case PoolingType::AVERAGE:
        return std::accumulate(flat.begin(), flat.end(), 0.0) / flat.size();
    default:
        throw std::runtime_error("Unknown pooling type.");
    }
}

std::vector<std::vector<std::vector<double>>> Pooling::forward(
    const std::vector<std::vector<std::vector<double>>> &input)
{
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    int out_h = (height - kernel_h) / stride + 1;
    int out_w = (width - kernel_w) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(
        channels,
        std::vector<std::vector<double>>(out_h, std::vector<double>(out_w, 0.0)));

    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < out_h; ++i)
            for (int j = 0; j < out_w; ++j)
            {
                std::vector<std::vector<double>> region(kernel_h, std::vector<double>(kernel_w, 0.0));
                for (int ki = 0; ki < kernel_h; ++ki)
                {
                    for (int kj = 0; kj < kernel_w; ++kj)
                    {
                        int row = i * stride + ki - padding;
                        int col = j * stride + kj - padding;

                        if (row >= 0 && row < height && col >= 0 && col < width)
                            region[ki][kj] = input[c][row][col];
                        else
                            region[ki][kj] = 0.0;
                    }
                }
                output[c][i][j] = pool_region(region);
            }
    return output;
}
