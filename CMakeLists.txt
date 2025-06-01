cmake_minimum_required(VERSION 3.10)
project(perceptron)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# Ejecutable de entrenamiento
add_executable(mlp_train
     main_train.cpp
     src/layer.cpp
     src/mlp.cpp
     src/neuron.cpp
     src/dataset.cpp
     src/config.cpp
)

# Ejecutable de prueba
add_executable(mlp_test
     main_test.cpp
     src/layer.cpp
     src/mlp.cpp
     src/neuron.cpp
     src/dataset.cpp
     src/config.cpp
)

add_executable(mnist
     main.cpp
     src/layer.cpp
     src/mlp.cpp
     src/neuron.cpp
     src/dataset.cpp
     src/config.cpp
)