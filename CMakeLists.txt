cmake_minimum_required(VERSION 3.10)
project(perceptron)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(conv2d
     main_conv2d.cpp
     src/conv2d.cpp
)

add_executable(main
     main.cpp
     src/conv2d.cpp
     src/flatten.cpp
     src/pooling.cpp
)