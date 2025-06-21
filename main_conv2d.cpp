#include "conv2d.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

std::vector<std::vector<std::vector<std::vector<double>>>> load_images_from_txt(
    const std::string &path_txt, int num_images, int channels, int height, int width)
{
    std::ifstream file(path_txt);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path_txt);
    std::vector<std::vector<std::vector<std::vector<double>>>> images;
    std::string line;
    int line_count = 0;
    std::vector<std::vector<std::vector<double>>> current_image(
        channels, std::vector<std::vector<double>>(height, std::vector<double>(width)));
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        double val;
        int idx = 0;
        int channel = line_count % channels;
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
            {
                if (!(ss >> val))
                    throw std::runtime_error("Error reading value at line " + std::to_string(line_count + 1));
                current_image[channel][i][j] = val;
            }
        line_count++;
        if (line_count % channels == 0)
        {
            images.push_back(current_image);
            if (images.size() >= static_cast<size_t>(num_images))
                break;
        }
    }
    if (images.size() != static_cast<size_t>(num_images))
        std::cerr << "Warning: Expected " << num_images << " images, but found " << images.size() << ".\n";
    file.close();
    return images;
}

void save_images_to_txt(
    const std::string &output_path,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &images)
{
    std::ofstream file(output_path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file for writing: " + output_path);
    std::cout << "Saving with image size " << images[0].size() << "x" << images[0][0].size() << "x" << images[0][0][0].size() << "\n";
    for (const auto &image : images)
    {
        for (const auto &channel : image)
        {
            for (const auto &row : channel)
            {
                for (double val : row)
                    file << val << " ";
                file << "\n";
            }
            file << "\n";
        }
    }
    file.close();
    std::cout << "Saved to " << output_path << ".\n";
}

int main()
{
    const int height = 28, width = 28;
    auto images = load_images_from_txt("./output/input.txt", 6, 3, height, width);
    std::cout << "Loaded image of size " << images[0].size() << "x" << images[0][0].size() << "x" << images[0][0][0].size() << "\n";
    Conv2D conv(3, 2, 3, 3, 1, 1); // in_channels=3, out_channels=2, kernel=3x3, stride=1, padding=1
    auto image_processed = images;
    for (size_t i = 0; i < image_processed.size(); ++i)
        image_processed[i] = conv.forward(images[i]);
    save_images_to_txt("./output/output_conv.txt", image_processed);
    return 0;
}
