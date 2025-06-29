# MLP (Multi-Layer Perceptron) - MNIST

By Braulio Nayap Maldonado Casilla

## Introducción

El **MLP (Perceptrón Multicapa)** es un tipo de red neuronal artificial compuesta por múltiples capas de neuronas organizadas jerárquicamente. A diferencia del perceptrón simple, el MLP puede aprender representaciones no lineales gracias a sus capas ocultas y al uso de funciones de activación no lineales como la sigmoide, la ReLU o la tangente hiperbólica. Estas características lo hacen especialmente adecuado para resolver problemas complejos como la clasificación no lineal, por ejemplo, el problema clásico de la compuerta XOR.

---

## Implementación en C++

### Clase Neuron

La clase `Neuron` representa una neurona individual dentro de una red neuronal. Está diseñada para ser flexible, permitiendo la inicialización de pesos aleatorios con un generador y distribución personalizados. Maneja un vector de pesos (`weights`) y un sesgo (`bias`), y provee métodos para serializar y deserializar su estado (guardar y cargar desde archivos o flujos de datos).

```cpp
class Neuron
{
private:
    std::vector<double> weights;
    double bias;

public:
    Neuron() = default;

    template <typename RNG, typename Dist>
    Neuron(int n_inputs, RNG &gen, Dist &dis);

    std::vector<double> &get_weights();
    double &get_bias();
    const std::vector<double> &get_weightss() const;
    const double &get_biass() const;

    void save(std::ostream &out) const;
    void load(std::istream &in, int n_inputs);
    void update(std::shared_ptr<Optimizer> optimizer, double learning_rate,
                const double *input, double delta,
                int input_size, int neuron_index, int layer_index);
    void compute_penalty(double &penalty) const;
};
```

#### Constructor por Defecto

Inicializa una neurona sin pesos ni sesgo definidos. Es útil cuando los pesos se cargarán más adelante mediante el método `load`.

```cpp
Neuron() = default;
```

#### Constructor Parametrizado con Generador Aleatorio

Este constructor inicializa la neurona con un número especificado de entradas (`n_inputs`). Cada peso se asigna aleatoriamente usando una distribución (`dis`) y un generador de números aleatorios (`gen`), lo que permite gran flexibilidad para pruebas o inicializaciones personalizadas (por ejemplo, normal, uniforme, etc.). El sesgo (`bias`) se inicializa a cero.

```cpp
template <typename RNG, typename Dist>
Neuron(int n_inputs, RNG &gen, Dist &dis)
    : weights(n_inputs)
{
    for (int i = 0; i < n_inputs; ++i)
        weights[i] = dis(gen);

    bias = 0.0;
}
```

#### Métodos de Acceso (GET)

- `get_weights` y `get_weightss` devuelven referencias a los pesos, para permitir lectura y modificación directa.

  ```cpp
  std::vector<double> &get_weights();                 // Versión modificable
  const std::vector<double> &get_weightss() const;    // Versión de solo lectura
  ```

- `get_bias` y `get_biass` devuelven referencias al sesgo.

  ```cpp
  double &get_bias();               // Versión modificable
  const double &get_biass() const; // Versión de solo lectura
  ```

#### Guardado de la Neurona

Este método escribe todos los pesos y el sesgo en un flujo de salida (`std::ostream`), separándolos por espacios. Es útil para guardar el estado de la neurona en un archivo de texto o binario.

```cpp
void Neuron::save(std::ostream &out) const
{
    for (const auto &w : weights)
        out << w << " ";
    out << bias << "\n";
}
```

#### Carga de la Neurona

Este método lee los pesos y el sesgo desde un flujo de entrada (`std::istream`). Es importante especificar cuántos pesos se esperan (`n_inputs`) para poder redimensionar adecuadamente el vector.

```cpp
void Neuron::load(std::istream &in, int n_inputs)
{
    weights.resize(n_inputs);
    for (int i = 0; i < n_inputs; ++i)
        in >> weights[i];
    in >> bias;
}
```

#### Actualización de la neurona

Este método actualiza pesos y sesgo usando un optimizador (`opt`). Recibe la tasa de aprendizaje (`learning_rate`), el vector de entrada (`input`), el error local (`delta`), tamaño de entrada (`input_size`), índice de la neurona (`neuron_index`) y de la capa (`layer_index`). Calcula un ID global para identificar la neurona y llama a `opt->update` para modificar los parámetros.

```cpp
void Neuron::update(std::shared_ptr<Optimizer> opt, double learning_rate,
                    const double *input, double delta,
                    int input_size, int neuron_index, int layer_index)
{
    int global_id = layer_index * 100000 + neuron_index;
    opt->update(learning_rate, weights, bias,
                input, delta, input_size, global_id);
}
```

#### Cálculo de Penalización (Regularización L2)

Este método acumula la penalización por regularización L2 para la neurona. Recorre cada peso (`weight`) y suma su cuadrado al valor acumulado de `penalty`. Esta penalización se utiliza típicamente para evitar el sobreajuste al modelo, forzando a los pesos a mantenerse pequeños.

```cpp
void Neuron::compute_penalty(double &penalty) const
{
    for (const auto &weight : weights)
        penalty += weight * weight;
}
```

---

### Clase Layer

La clase `Layer` representa una capa de una red neuronal multicapa. Cada capa contiene un conjunto de neuronas (`Neuron`) y una función de activación asociada. Esta clase se encarga de inicializar, propagar hacia adelante (forward pass) de forma lineal y aplicar la activación, así como de guardar y cargar sus parámetros desde archivos.

```cpp
class Layer
{
private:
    int input_size;
    int output_size;
    std::vector<Neuron> neurons;
    ActivationType activation;

public:
    Layer(int in_size, int out_size, ActivationType act);
    Layer(int in_size, int out_size, ActivationType act, bool true_random);
    void linear_forward(const std::vector<double> &input, std::vector<double> &output);
    void apply_activation(std::vector<double> &output) const;
    void apply_update(std::shared_ptr<Optimizer> optimizer,
                      const std::vector<double> &delta,
                      const std::vector<double> &input,
                      double learning_rate, int layer_index);
    // Getters
    int get_input_size();
    int get_output_size();
    const int get_inputss() const;
    const int get_outputss() const;
    std::vector<Neuron> &get_neurons();
    const std::vector<Neuron> &get_neuronss() const;
    ActivationType get_activation() const;
    const int get_neurons_size() const;

    // I/O
    void save(std::ostream &out, const int i) const;
    void load(std::istream &in);
    void compute_penalty(double &penalty) const;
};
```

#### Constructores

##### Constructor con inicialización aleatoria

Este constructor crea una capa con `output_size` neuronas, cada una con `input_size` entradas. Los pesos se inicializan aleatoriamente con una distribución uniforme en el rango [-limit, limit], donde:

![Funcion Limite](.docs/f1.png)

Esto mejora la inicialización para evitar problemas de desvanecimiento del gradiente.

```cpp
Layer(int in_size, int out_size, ActivationType act)
    : input_size(in_size), output_size(out_size), activation(act)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::uniform_real_distribution<> dis(-limit, limit);

    for (int i = 0; i < output_size; ++i)
        neurons.emplace_back(input_size, gen, dis);
}
```

##### Constructor sin inicialización aleatoria

Este constructor crea una capa con neuronas vacías, útil cuando se desea cargar los pesos desde un archivo.

```cpp
Layer(int in_size, int out_size, ActivationType act, bool true_random)
    : input_size(in_size), output_size(out_size), activation(act)
{
    neurons.resize(output_size, Neuron());
}
```

#### Propagación hacia Adelante (linear_forward)

El método `linear_forward` aplica la operación de propagación hacia adelante en una capa, realizando la multiplicación de cada neurona:

![Funcion forward](.docs/f2.png)

En la funcón `apply_activation`.

```cpp
void Layer::linear_forward(const std::vector<double> &input, std::vector<double> &output)
{
    for (int i = 0; i < output_size; ++i)
    {
        output[i] = neurons[i].get_biass();
        for (int j = 0; j < input_size; ++j)
        {
            output[i] += input[j] * neurons[i].get_weightss()[j];
        }
    }

    if (activation == SIGMOID)
        for (double &val : output) val = sigmoid(val);
    else if (activation == RELU)
        for (double &val : output) val = relu(val);
    else if (activation == SOFTMAX)
        softmax(output, output);
    else if (activation == TANH)
        for (double &val : output) val = tanh(val);
}

void Layer::apply_activation(std::vector<double> &output) const
{
    if (this->activation == SIGMOID)
    {
        for (double &val : output)
            val = sigmoid(val);
    }
    else if (this->activation == RELU)
    {
        for (double &val : output)
            val = relu(val);
    }
    else if (this->activation == TANH)
    {
        for (double &val : output)
            val = tanh(val);
    }
    else if (this->activation == SOFTMAX)
    {
        softmax(output, output);
    }
}
```

#### Aplicar actualización en la capa

Este método aplica la actualización a todos los `neurons` de la capa usando un optimizador (`optimizer`). Recibe el vector de errores locales (`delta`), entradas (`input`), tasa de aprendizaje (`learning_rate`) y el índice de la capa (`layer_index`). Itera sobre las neuronas y llama a `update` en cada una, pasando los datos correspondientes.

```cpp
void Layer::apply_update(std::shared_ptr<Optimizer> optimizer, const std::vector<double> &delta,
                         const std::vector<double> &input,
                         double learning_rate, int layer_index)
{
    for (int i = 0; i < this->output_size; ++i)
        neurons[i].update(optimizer, learning_rate, input.data(), delta[i], this->input_size, i, layer_index);
}
```

#### Métodos de Guardado y Carga

Permiten serializar y deserializar la información de las neuronas de la capa. Se guarda cada neurona con su índice de capa y de posición:

```cpp
void Layer::save(std::ostream &out, const int i) const
{
    for (size_t j = 0; j < neurons.size(); ++j)
    {
        out << i + 1 << " " << j + 1 << " ";
        neurons[j].save(out);
    }
}
```

```cpp
void Layer::load(std::istream &in)
{
    neurons.resize(output_size);
    for (int j = 0; j < output_size; ++j)
    {
        int layer_idx, neuron_idx;
        in >> layer_idx >> neuron_idx;
        neurons[j].load(in, input_size);
    }
}
```

#### Métodos GET

Permiten acceder a información relevante de la capa, incluyendo el número de entradas y salidas, el tipo de activación y la lista de neuronas:

```cpp
int get_input_size();                  // Devuelve la cantidad de entradas
int get_output_size();                 // Devuelve la cantidad de salidas
const std::vector<Neuron> &get_neuronss() const; // Neuronas (solo lectura)
std::vector<Neuron> &get_neurons();    // Neuronas (modificable)
ActivationType get_activation() const; // Tipo de activación usado
```

#### Cálculo de Penalización en la Capa

Este método acumula la penalización por regularización L2 de todos los pesos de la capa. Para ello, llama a `compute_penalty` de cada neurona que la compone, sumando sus respectivas penalizaciones al valor total de `penalty`.

```cpp
void Layer::compute_penalty(double &penalty) const
{
    for (const auto &neuron : neurons)
        neuron.compute_penalty(penalty);
}
```

---

### Clase Mlp

La clase `Mlp` (Multilayer Perceptron) representa una **red neuronal multicapa** compuesta por múltiples capas (`Layer`), cada una con su propio conjunto de neuronas y función de activación. Esta clase maneja el _forward pass_, el _backward pass_, el entrenamiento, evaluación, guardado y carga del modelo, orientada a tareas de clasificación como el dataset MNIST.

```cpp
class Mlp
{
private:
    int n_inputs;
    int n_outputs;
    double learning_rate;
    std::vector<Layer> layers;
    std::shared_ptr<Optimizer> optimizer = nullptr;
    std::shared_ptr<Regularizer> regularizer = nullptr;
    std::shared_ptr<DropoutController> dropout = nullptr;

public:
    Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
        double lr, std::vector<ActivationType> activation_types,
        optimizer_type opt = optimizer_type::SGD,
        bool regularizer = false, bool dropout = false);
    Mlp() = default;
    void forward(const std::vector<double> &input, std::vector<std::vector<double>> &activations, bool train);
    void backward(const std::vector<double> &input,
                  const std::vector<std::vector<double>> &activations,
                  const std::vector<double> &expected);
    void one_hot_encode(int label, std::vector<double> &target);
    double cross_entropy_loss(const std::vector<double> &predicted,
                             const std::vector<double> &expected);
    void train(std::vector<std::vector<double>> &images,
                std::vector<int> & labels,double &average_loss,
                double &train_accuracy);
    void test(const std::vector<std::vector<double>> &images,
             const std::vector<int> &labels, double &test_accuracy);
    void train_test(std::vector<std::vector<double>> &train_images,
                    std::vector<int> &train_labels,
                    const std::vector<std::vector<double>> &test_images, const std::vector<int> &test_labels,
                    bool Test, const std::string &dataset_filename, int epochs = 1000);
    void save_data(const std::string &filename) const;
    bool load_data(const std::string &filename);
    void test_info(const std::vector<std::vector<double>> &X_test, const std::vector<int> &y_test);
};
```

---

#### Constructor

Este constructor crea e inicializa una red neuronal multicapa (MLP), configurando su arquitectura, tipo de optimizador, y técnicas adicionales de regularización como L2 y Dropout. Recibe como parámetros el número de entradas (`n_inputs`), una lista con el tamaño de cada capa oculta (`layer_sizes`), el número de salidas (`n_outputs`), la tasa de aprendizaje (`lr`), las funciones de activación para cada capa (`activation_types`), el tipo de optimizador (`opt`), un indicador para usar regularización L2 (`regularizer`) y otro para habilitar Dropout (`dropout`).

Primero, el constructor verifica que el número de capas ocultas coincida con la cantidad de funciones de activación proporcionadas. Luego, construye las capas de la red (`Layer`) una por una, conectándolas secuencialmente. Después se inicializa el optimizador seleccionado, que puede ser RMSProp, Adam o SGD.

Si se activa la regularización, se crea un objeto `L2Regularizer` con un valor por defecto (0.01) y se asigna al optimizador. Si se habilita `Dropout`, se inicializa un controlador con una tasa predefinida (0.5), que puede usarse durante el entrenamiento para apagar aleatoriamente ciertas neuronas y reducir el sobreajuste.

```cpp
Mlp::Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
         double lr, std::vector<ActivationType> activation_types, optimizer_type opt,
         bool regularizer, bool dropout)
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

    if (opt == optimizer_type::RMSPROP)
        this->optimizer = std::make_shared<RMSProp>();
    else if (opt == optimizer_type::ADAM)
        this->optimizer = std::make_shared<Adam>();
    else
        this->optimizer = std::make_shared<SGD>();

    if (regularizer)
    {
        double l2_penalty = 0.01;
        this->regularizer = std::make_shared<L2Regularizer>(l2_penalty);
        this->optimizer->set_regularizer(this->regularizer);
        std::cout << "Using L2 regularization: " << l2_penalty << ".\n";
    }

    if (dropout)
    {
        double dropout_rate = 0.5;
        this->dropout = std::make_shared<DropoutController>(0.5);
        std::cout << "Using dropout: " << dropout_rate << ".\n";
    }
}
```

#### `forward`

La función `forward` realiza la propagación hacia adelante en la red neuronal, calculando las salidas capa por capa a partir de una entrada. Comienza limpiando el vector de activaciones y almacenando la entrada original como la primera activación. Luego, para cada capa, se calcula la salida lineal con `linear_forward` y se aplica la función de activación correspondiente mediante `apply_activation`.

Si está habilitado el mecanismo de _dropout_ (`dropout != nullptr`) y se está en modo entrenamiento (`train == true`), se aplica _dropout_ a las salidas de las capas ocultas (no a la de salida), desconectando aleatoriamente algunas neuronas para reducir el sobreajuste. Finalmente, cada vector de salida se almacena en la lista de activaciones, que contiene el estado completo de la red tras la propagación.

```cpp
void Mlp::forward(const std::vector<double> &input, std::vector<std::vector<double>> &activations, bool train)
{
    activations.clear();
    activations.push_back(input);

    for (size_t i = 0; i < layers.size(); ++i)
    {
        auto &layer = layers[i];
        std::vector<double> output(layer.get_output_size());
        layer.linear_forward(activations.back(), output);
        layer.apply_activation(output);
        if (dropout && train && i < layers.size() - 1)
            dropout->apply(output);
        activations.push_back(output);
    }
}
```

#### `backward`

Implementa la retropropagación del error desde la capa de salida hacia las capas anteriores, calculando los deltas (gradientes) para cada neurona según la diferencia entre la salida actual y la esperada, ajustada por la derivada de la función de activación. Luego, utiliza estos deltas para actualizar los pesos y sesgos de cada neurona, aplicando la función asociada al optimizador.

```cpp
void backward(const std::vector<double> &input,
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


    if (this->optimizer == nullptr)
    {
        std::cerr << "Error: optimizer is nullptr \n";
        exit(1);
    }

    for (size_t l = 0; l < layers.size(); ++l)
        layers[l].apply_update(this->optimizer, deltas[l], activations[l], learning_rate, l);
}
```

#### `one_hot_encode`

Convierte una clase (por ejemplo, `3`) a un vector one-hot (por ejemplo, `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`), para facilitar la representación de etiquetas en problemas de clasificación.

```cpp
void one_hot_encode(int label, std::vector<double> &target)
{
    std::fill(target.begin(), target.end(), 0.0);
    target[label] = 1.0;
}
```

#### `cross_entropy_loss`

Calcula la pérdida entre la salida predicha y la esperada usando la función de entropía cruzada, sumando el producto negativo de la etiqueta esperada por el logaritmo de la probabilidad predicha, con un pequeño valor `epsilon` para evitar errores numéricos, siendo ideal para problemas de clasificación multiclase con salida softmax.

```cpp
double cross_entropy_loss(const std::vector<double> &predicted, const std::vector<double> &expected)
{
    double loss = 0.0;
    const double epsilon = 1e-9;
    for (size_t i = 0; i < predicted.size(); ++i)
        loss -= expected[i] * log(predicted[i] + epsilon);
    return loss;
}
```

#### `train`

Entrena la red neuronal utilizando un conjunto de imágenes y sus respectivas etiquetas. Para cada muestra, el método realiza un pase hacia adelante con activación y _dropout_ (si está habilitado), calcula la pérdida usando entropía cruzada, y luego realiza retropropagación para ajustar los pesos.

Durante el entrenamiento, se acumulan tanto la pérdida total como el número de predicciones correctas. Si hay regularización activada (por ejemplo, L2), se calcula la penalización asociada a los pesos de la red y se añade al total de pérdida. Al final del ciclo, se actualiza la pérdida promedio (`average_loss`) y la precisión de entrenamiento (`train_accuracy`).

```cpp
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
    double penalty = 0.0;
    int correct = 0;

    for (size_t k = 0; k < n; ++k)
    {
        size_t i = indices[k];
        const auto &input = images[i];
        int label = labels[i];
        one_hot_encode(label, target);
        forward(input, activations, true);
        total_loss += cross_entropy_loss(activations.back(), target);
        backward(input, activations, target);

        int pred = std::distance(activations.back().begin(),
                                 std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == label)
            ++correct;
    }
    if (regularizer)
        penalty = regularizer->compute_penalty(layers);
    average_loss = (total_loss + penalty) / n;
    train_accuracy = 100.0 * correct / n;
}
```

#### `test`

Evalúa el desempeño de la red neuronal sobre un conjunto de datos sin alterar los pesos del modelo. Para cada muestra, realiza una propagación hacia adelante con `train = false`, lo cual asegura que no se aplique _dropout_ u otras técnicas específicas del entrenamiento. Luego compara la predicción con la etiqueta real y contabiliza los aciertos.

Al finalizar, calcula la precisión (`test_accuracy`) como el porcentaje de predicciones correctas sobre el total de muestras evaluadas.

```cpp
void Mlp::test(const std::vector<std::vector<double>> &images, const std::vector<int> &labels, double &test_accuracy)
{
    int correct = 0;
    std::vector<std::vector<double>> activations;

    for (size_t i = 0; i < images.size(); ++i)
    {
        forward(images[i], activations, false);
        int pred = std::distance(activations.back().begin(),
                                 std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == labels[i])
            ++correct;
    }

    test_accuracy = 100.0 * correct / images.size();
}
```

#### `train_test`

Gestiona el ciclo completo de entrenamiento y evaluación de la red durante un número determinado de épocas, ejecutando en cada época el entrenamiento con el conjunto de datos de entrenamiento y, opcionalmente, la evaluación con el conjunto de prueba para medir la precisión. Además, registra el progreso en un archivo de log, muestra estadísticas por consola, guarda el modelo periódicamente, y el mejor modelo según el accuracy del test y detiene el proceso anticipadamente si se cumplen criterios de convergencia o precisión deseada.

```cpp
void train_test(std::vector<std::vector<double>> &train_images, std::vector<int> &train_labels,
                     const std::vector<std::vector<double>> &test_images, const std::vector<int> &test_labels,
                     bool Test, const std::string &dataset_filename, int epochs=1000)
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
    double best_test_accuracy = -1.0;
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
        {
            log_line << ", Test Acc: " << test_accuracy << "%"
                     << ", Test Time: " << test_time << "s";
            if (test_accuracy > best_test_accuracy)
            {
                best_test_accuracy = test_accuracy;
                std::string best_model_path = (output_dir / "best_model.dat").string();
                save_data(best_model_path);
            }
        }

        std::cout << log_line.str() << std::endl;
        log_file << log_line.str() << std::endl;
        if ((epoch + 1) % 10 == 0)
        {
            std::string filename = (output_dir / ("epoch_" + std::to_string(epoch + 1) + ".dat")).string();
            save_data(filename);
            std::cout << "Model saved at epoch " << (epoch + 1) << " to " << filename << ".\n";
        }
        if (average_loss < 0.00001 || epoch >= epochs)
        {
            std::cout << "Stopping training: early stopping criteria met.\n";
            break;
        }
        epoch++;
    }
}
```

#### `save_data`

Guarda en un archivo el estado completo del modelo, incluyendo la arquitectura (número de entradas, neuronas por capa), tasa de aprendizaje, funciones de activación y los pesos y sesgos de cada neurona. Este proceso se realiza delegando la serialización específica a cada capa mediante su método `save`.

```cpp
void save_data(const std::string &filename) const
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

    out << to_string(optimizer->get_type()) << "\n";

    // Capas y neuronas
    for (size_t i = 0; i < layers.size(); ++i)
        layers[i].save(out, i);
    out.close();
}
```

#### `load_data`

Carga la configuración completa del modelo desde un archivo, reconstruyendo la arquitectura (número de entradas, neuronas por capa), la tasa de aprendizaje y las funciones de activación. Luego, crea las capas correspondientes y recupera los pesos y sesgos de cada neurona llamando a su método `load`, dejando el modelo listo para usarse.

```cpp
bool load_data(const std::string &filename)
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
    // Leer optimizador
    std::string opt_type;
    in >> opt_type;
    optimizer_type opt = from_string_opt(opt_type);
    if (opt == optimizer_type::RMSPROP)
        this->optimizer = std::make_shared<RMSProp>();
    else if (opt == optimizer_type::ADAM)
        this->optimizer = std::make_shared<Adam>();
    else
        this->optimizer = std::make_shared<SGD>();

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
```

#### `test_info`

Evalúa la red y muestra detalles como las predicciones vs etiquetas reales para fines de diagnóstico o visualización de errores.

```cpp
void test_info(const std::vector<std::vector<double>> &X_test, const std::vector<int> &y_test)
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
```

---

### Clase Dataset

La clase `Dataset` se encarga de almacenar y manejar conjuntos de datos para el entrenamiento y prueba de redes neuronales. Soporta la carga de archivos de texto con datos estructurados (como los generados desde MNIST) y provee funciones para acceder y visualizar el conjunto de datos.

```cpp
class Dataset
{
private:
    std::vector<std::vector<double>> X; // Vectores de entrada
    std::vector<std::vector<double>> y; // Vectores de salida

public:
    Dataset(const std::string &filename);
    Dataset() = default;
    const std::vector<std::vector<double>> &get_X() const;
    const std::vector<std::vector<double>> &get_y() const;
    const std::vector<int> &get_ys() const;
    void print_data(std::string name) const;
    void generate_mnist(const std::string &filename, int TRAIN_SAMPLES, int TEST_SAMPLES);
};
```

#### Constructor con archivo

Este constructor carga un dataset desde un archivo de texto con el siguiente formato:

- Primera línea: dos enteros `n_inputs` y `n_outputs`.
- Líneas siguientes: `n_inputs` valores reales seguidos de `n_outputs` valores (en one-hot).

```cpp
Dataset::Dataset(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cant open file: " + filename);

    int n_inputs, n_outputs;
    file >> n_inputs >> n_outputs;
    if (n_inputs < 1 || n_outputs < 1)
        throw std::runtime_error("Invalid number of inputs or options in file");

    X.clear();
    y.clear();

    std::vector<double> input_row(n_inputs);
    std::vector<double> output_row(n_outputs);

    while (file)
    {
        for (int i = 0; i < n_inputs; ++i)
            if (!(file >> input_row[i]))
                return;

        for (int j = 0; j < n_outputs; ++j)
            if (!(file >> output_row[j]))
                throw std::runtime_error("Unexpected end of file while reading output vector");

        X.push_back(input_row);
        y.push_back(output_row);
    }

    file.close();
}
```

#### Métodos GET

- `get_X`: Devuelve todos los vectores de entrada (features).

  ```cpp
  const std::vector<std::vector<double>> &Dataset::get_X() const
  {
      return X;
  }
  ```

- `get_y`: Devuelve todos los vectores de salida (normalmente en one-hot).

  ```cpp
  const std::vector<std::vector<double>> &Dataset::get_y() const
  {
      return y;
  }
  ```

- `get_ys`: Devuelve un vector con la clase esperada de cada muestra (entero entre 0 y 9 si es MNIST), identificada como la posición del mayor valor en el vector de salida (útil para evaluación).

  ```cpp
  const std::vector<int> &Dataset::get_ys() const
  {
      static std::vector<int> ys;
      ys.clear();
      for (const auto &output_row : y)
      {
          int max_index = std::distance(output_row.begin(), std::max_element(output_row.begin(), output_row.end()));
          ys.push_back(max_index);
      }
      return ys;
  }
  ```

#### Función `print_data`

Imprime por consola estadísticas del conjunto cargado, incluyendo nombre, número de muestras, número de entradas y salidas por muestra.

```cpp
void Dataset::print_data(std::string name) const
{
    std::cout << "==========================\n";
    std::cout << "DATASET " << name << "\n";
    std::cout << "\tTotal samples: " << X.size() << "\n";
    std::cout << "\tInputs per sample: " << (X.empty() ? 0 : X[0].size()) << "\n";
    std::cout << "\tOutputs per sample: " << (y.empty() ? 0 : y[0].size()) << "\n";
    std::cout << "==========================\n\n";
}
```

#### Función `generate_mnist`

Esta función lee los archivos binarios originales del dataset **MNIST** y genera archivos `.txt` legibles para entrenamiento/prueba.

##### `reverse_int`

Convierte enteros en formato _big-endian_ (como los usa MNIST) a _little-endian_ (como los usa la mayoría de las plataformas modernas).

```cpp
int reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
```

##### `read_mnist_images`

Lee imágenes MNIST y normaliza los píxeles en `[0, 1]`.

```cpp
void read_mnist_images(const std::string &filename, std::vector<std::vector<double>> &images, int num_images)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file " + filename);
    int magic, n, rows, cols;
    file.read((char *)&magic, 4);
    magic = reverse_int(magic);
    file.read((char *)&n, 4);
    n = reverse_int(n);
    file.read((char *)&rows, 4);
    rows = reverse_int(rows);
    file.read((char *)&cols, 4);
    cols = reverse_int(cols);
    images.resize(num_images, std::vector<double>(rows * cols));
    for (int i = 0; i < num_images; ++i)
        for (int j = 0; j < rows * cols; ++j)
        {
            unsigned char pixel;
            file.read((char *)&pixel, 1);
            images[i][j] = pixel / 255.0;
        }
}
```

##### `read_mnist_labels`

Lee las etiquetas asociadas a las imágenes.

```cpp
void read_mnist_labels(const std::string &filename, std::vector<int> &labels, int num_labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file " + filename);
    int magic, n;
    file.read((char *)&magic, 4);
    magic = reverse_int(magic);
    file.read((char *)&n, 4);
    n = reverse_int(n);
    labels.resize(num_labels);
    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char label;
        file.read((char *)&label, 1);
        labels[i] = label;
    }
}
```

#### Formato de los archivos generados:

- Primera línea: número de entradas (784) y número de salidas (10).
- Líneas siguientes: 784 valores normalizados de píxeles (0–1), seguidos por 10 valores en one-hot según la etiqueta.

```cpp
void Dataset::generate_mnist(const std::string &filename, int TRAIN_SAMPLES, int TEST_SAMPLES)
{
    std::vector<std::vector<double>> train_images, test_images;
    std::vector<int> train_labels, test_labels;

    // Leer imágenes y etiquetas de MNIST
    read_mnist_images("./archive/train-images.idx3-ubyte", train_images, TRAIN_SAMPLES);
    read_mnist_labels("./archive/train-labels.idx1-ubyte", train_labels, TRAIN_SAMPLES);
    read_mnist_images("./archive/t10k-images.idx3-ubyte", test_images, TEST_SAMPLES);
    read_mnist_labels("./archive/t10k-labels.idx1-ubyte", test_labels, TEST_SAMPLES);

    std::ofstream train_file(filename + "train.txt");
    std::ofstream test_file(filename + "test.txt");

    if (!train_file || !test_file)
        throw std::runtime_error("No se pudieron crear archivos de salida train/test");

    int n_inputs = train_images[0].size();
    int n_outputs = 10;

    train_file << n_inputs << " " << n_outputs << "\n";
    test_file << n_inputs << " " << n_outputs << "\n";

    for (size_t i = 0; i < train_images.size(); ++i)
    {
        for (double v : train_images[i])
            train_file << v << " ";
        for (int j = 0; j < n_outputs; ++j)
            train_file << (train_labels[i] == j ? 1 : 0) << " ";
        train_file << "\n";
    }

    for (size_t i = 0; i < test_images.size(); ++i)
    {
        for (double v : test_images[i])
            test_file << v << " ";
        for (int j = 0; j < n_outputs; ++j)
            test_file << (test_labels[i] == j ? 1 : 0) << " ";
        test_file << "\n";
    }

    train_file.close();
    test_file.close();
}
```

---

### Clase Config

La clase `Config` permite cargar desde un archivo de texto la configuración de una red neuronal multicapa, incluyendo el número de capas, el tamaño de cada capa, las funciones de activación de cada capa y la tasa de aprendizaje (`learning_rate`). Esto permite construir arquitecturas de redes fácilmente ajustables sin recompilar el programa.

- `n_inputs`: Número de entradas a la red (definido externamente al archivo de configuración).
- `layer_sizes`: Tamaños de todas las capas (ocultas + salida).
- `learning_rate`: Tasa de aprendizaje usada en el entrenamiento.
- `activations`: Funciones de activación por capa, una por cada tamaño en `layer_sizes`.

```cpp
class Config
{
private:
    int n_inputs;
    std::vector<int> layer_sizes;
    float learning_rate;
    std::vector<ActivationType> activations;
    optimizer_type opt;

public:
    bool load_config(const std::string &filename, const int inputs);
    const std::vector<int> &get_layer_sizes() const;
    float get_learning_rate() const;
    const std::vector<ActivationType> &get_activations() const;
    const optimizer_type &get_optimizer() const;
    const void print_config();
};
```

#### Método `load_config`

Carga los parámetros desde un archivo de texto siguiendo este formato:

```
[capas]          // línea 1: tamaños de capa separados por espacio (ej: 32 16 10)
[lr]             // línea 2: tasa de aprendizaje (ej: 0.01)
[activaciones]   // línea 3: nombres de funciones por capa (ej: relu relu softmax)
[optimizador]    // linea 4: nombre de optimizador (ej: sgd, rmsprop o adam)
```

```cpp
bool Config::load_config(const std::string &filename, const int inputs)
```

#### Métodos GET

- `get_layer_sizes`: Devuelve los tamaños de las capas.

  ```cpp
  const std::vector<int> &get_layer_sizes() const;
  ```

- `get_learning_rate`: Devuelve la tasa de aprendizaje.

  ```cpp
  float get_learning_rate() const;
  ```

- `get_activations`: Devuelve las funciones de activación como `ActivationType`.

  ```cpp
  const std::vector<ActivationType> &get_activations() const;
  ```

- `get_optimizer`: Devuelve las funciones de activación como `optimizer_type`.

  ```cpp
  const optimizer_type &get_optimizer() const;
  ```

---

#### Método `print_config`

Imprime en consola los parámetros cargados, útil para verificar la configuración antes de inicializar la red neuronal:

```cpp
const void Config::print_config()
{
    std::cout << "==========================\n";
    std::cout << "CONFIGURATION\n";
    std::cout << "Layer Sizes: \n";
    std::cout << "\t" << n_inputs << " inputs\n";
    for (const auto &size : layer_sizes)
        std::cout << "\t" << size << " neurons\n";
    std::cout << "Learning Rate: " << learning_rate << "\n";
    std::cout << "Activations: \n";
    for (const auto &act : activations)
        std::cout << "\t" << to_string(act) << "\n";
    std::cout << "Optimizer: " << to_string(opt) << "\n";
    std::cout << "==========================\n";
    std::cout << std::endl;
}
```

---

### Archivo `activation.hpp`

Este archivo define las funciones de activación utilizadas en una red neuronal, sus derivadas, y herramientas auxiliares para trabajar con ellas. Permite usar diferentes funciones de activación para las capas (como `sigmoid`, `relu`, `tanh` o `softmax`) de forma modular, además de convertir entre `string` y `enum` para facilitar la configuración desde archivos externos.

#### Enum `ActivationType

```cpp
enum ActivationType
{
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};
```

#### Funciones de Activación

- Sigmoid

Suaviza los valores a un rango entre 0 y 1. Ideal para problemas binarios.
Su derivada se calcula como `x(1 - x)`, asumiendo que `x` ya es la salida de `sigmoid(x_original)`.

```cpp
inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}
```

- ReLU

Rectified Linear Unit. Pasa los valores positivos y trunca los negativos a 0.
Su derivada se define: Es 1 si `x > 0`, 0 en caso contrario. Eficiente y útil en capas ocultas.

```cpp
inline double relu(double x)
{
    return x > 0 ? x : 0;
}

inline double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}
```

- Tanh

Función hiperbólica tangente. Salida entre -1 y 1.
Derivada de `tanh`, usando la identidad `1 - \tanh^2(x)`.

```cpp
inline double tanh_fn(double x)
{
    return std::tanh(x);
}

inline double tanh_derivative(double x)
{
    double t = std::tanh(x);
    return 1.0 - t * t;
}
```

- Softmax (Función de Activación para Vectores)

Transforma un vector en una distribución de probabilidad (valores entre 0 y 1 que suman 1).
Resta el valor máximo (`max_val`) para evitar problemas de overflow numérico.
Solo se usa al final de una red neuronal de clasificación multiclase.

```cpp
inline void softmax(const std::vector<double> &input, std::vector<double> &output)
{
    double max_val = *max_element(input.begin(), input.end());
    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    for (double &val : output)
        val /= sum;
}
```

#### Conversión `enum <→> string`

Se usan `unordered_map` para traducir entre nombres de funciones (`string`) y su tipo (`ActivationType`):

```cpp
inline const std::unordered_map<ActivationType, std::string> activation_to_string = { ... };
inline const std::unordered_map<std::string, ActivationType> string_to_activation = { ... };
```

- Estos mapas permiten cargar funciones desde un archivo de configuración fácilmente.

#### Función `to_string`

Convierte un `ActivationType` a su nombre (`"sigmoid"`, `"relu"`, etc.).

```cpp
inline std::string to_string(ActivationType type)
```

#### Función `from_string`

Convierte una cadena a su `ActivationType` correspondiente (ej. `"tanh"` → `TANH`).

```cpp
inline ActivationType from_string(const std::string &name)
```

---

### Archivo `optimizer.hpp`

Este archivo define las clases base y derivadas para los optimizadores usados en la actualización de pesos de una red neuronal. Incluye tres optimizadores principales: `SGD`, `RMSProp` y `Adam`. Cada uno implementa la función `update` para modificar los pesos y sesgo de una neurona dado un gradiente, tasa de aprendizaje y contexto (como el índice de la neurona). Además, se proveen funciones para convertir entre el tipo enumerado `optimizer_type` y su representación en `string`, facilitando la configuración externa.

#### Enum class

`enum class optimizer_type` define un conjunto cerrado y nombrado de constantes que representan los tipos de optimizadores disponibles. Usar `enum class` en vez de un `enum` tradicional evita colisiones de nombres, ya que el tipo es fuerte y se accede con el prefijo `optimizer_type::`. Esto mejora la seguridad y claridad del código.

```cpp
enum class optimizer_type
{
    SGD,
    ADAM,
    RMSPROP
};
```

#### Clase base `Optimizer`

La clase abstracta `Optimizer` define la interfaz general para los algoritmos de optimización que actualizan los pesos y el sesgo de una neurona durante el entrenamiento. Incluye el método virtual puro `update`, que debe ser implementado por todas las clases derivadas (como `SGD`, `Adam`, o `RMSProp`), permitiendo actualizar los parámetros en función del gradiente y el tipo de optimizador.

Además, esta versión incorpora un mecanismo opcional de regularización L2 mediante un puntero a un objeto `Regularizer`. El método `set_regularizer` permite establecer esta instancia desde fuera de la clase, proporcionando flexibilidad para aplicar penalizaciones sobre los pesos durante la actualización. El método `get_type` devuelve el tipo del optimizador, facilitando la identificación y la lógica dependiente del tipo.

```cpp
class Optimizer
{
protected:
    std::shared_ptr<Regularizer> regularizer = nullptr;

public:
    virtual void update(double learning_rate, std::vector<double> &weights, double &bias,
                        const double *input, double delta,
                        int input_size, int neuron_index) = 0;

    virtual optimizer_type get_type() const = 0;
    void set_regularizer(std::shared_ptr<Regularizer> reg)
    {
        this->regularizer = reg;
    }

    virtual ~Optimizer() = default;
};
```

#### Clase `SGD` (Descenso de Gradiente Estocástico)

`SGD` implementa el algoritmo de descenso de gradiente estocástico, que actualiza cada peso restando el gradiente del error multiplicado por la tasa de aprendizaje. En esta versión, se incluye soporte para regularización L2, sumando una penalización proporcional al valor del peso antes de aplicar la actualización.

![](.docs/f3.png)

```cpp
class SGD : public Optimizer
{
public:
    void update(double learning_rate, std::vector<double> &weights, double &bias,
                const double *input, double delta,
                int input_size, int) override
    {
        double lambda = (this->regularizer ? this->regularizer->lambda_value() : 0.0);

        for (int j = 0; j < input_size; ++j)
        {
            double grad = delta * input[j];
            if (lambda > 0.0)
                grad += lambda * weights[j];  // Aplicar penalización L2
            weights[j] -= learning_rate * grad;
        }

        bias -= learning_rate * delta;
    }

    optimizer_type get_type() const override { return optimizer_type::SGD; }
};
```

#### Clase `RMSProp`

`RMSProp` (Root Mean Square Propagation) mejora a SGD adaptando de forma individual la tasa de aprendizaje de cada peso utilizando una media móvil de los cuadrados de los gradientes. Esto permite que pesos con gradientes grandes se actualicen más lentamente, estabilizando el entrenamiento. Además, esta versión incorpora soporte opcional para regularización L2.

![](.docs/f4.png)

```cpp
class RMSProp : public Optimizer
{
private:
    double tau;
    double epsilon;
    std::unordered_map<int, std::vector<double>> r_w;
    std::unordered_map<int, double> r_b;

public:
    RMSProp(double tau = 0.99, double epsilon = 1e-8)
        : tau(tau), epsilon(epsilon) {}

    void update(double learning_rate, std::vector<double> &weights, double &bias,
                const double *input, double delta, int input_size, int neuron_index) override
    {
        auto &r_weights = r_w[neuron_index];
        auto &r_bias = r_b[neuron_index];

        if (r_weights.size() != static_cast<size_t>(input_size))
            r_weights.assign(input_size, 0.0);

        double lambda = (this->regularizer ? this->regularizer->lambda_value() : 0.0);

        for (int j = 0; j < input_size; ++j)
        {
            double grad = delta * input[j];
            if (lambda > 0.0)
                grad += lambda * weights[j];  // Regularización L2

            r_weights[j] = tau * r_weights[j] + (1.0 - tau) * (grad * grad);
            weights[j] -= learning_rate * grad / (std::sqrt(r_weights[j]) + epsilon);
        }

        double grad_b = delta;
        r_bias = tau * r_bias + (1.0 - tau) * (grad_b * grad_b);
        bias -= learning_rate * grad_b / (std::sqrt(r_bias) + epsilon);
    }

    optimizer_type get_type() const override { return optimizer_type::RMSPROP; }
};
```

#### Clase `Adam`

`Adam` (Adaptive Moment Estimation) combina las ventajas de **RMSProp** y **momentum**, manteniendo dos momentos:

- el primero: la media móvil de los gradientes (acelera),
- el segundo: la media móvil de los gradientes al cuadrado (normaliza la escala).

Además, se aplican correcciones de sesgo para compensar la inicialización en cero de los momentos. Esta implementación también soporta regularización L2 opcional, lo que permite un control adicional sobre el sobreajuste.

![](.docs/f5.png)

```cpp
class Adam : public Optimizer
{
private:
    double beta1;
    double beta2;
    double epsilon;
    std::unordered_map<int, std::vector<double>> m_w, v_w;
    std::unordered_map<int, double> m_b, v_b;
    std::unordered_map<int, int> timestep;
    std::unordered_map<int, double> beta1_pow_t;
    std::unordered_map<int, double> beta2_pow_t;

public:
    Adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    void update(double learning_rate, std::vector<double> &weights, double &bias,
                const double *input, double delta, int input_size, int neuron_index) override
    {
        auto &m_weights = m_w[neuron_index];
        auto &v_weights = v_w[neuron_index];
        auto &m_bias = m_b[neuron_index];
        auto &v_bias = v_b[neuron_index];
        auto &t = timestep[neuron_index];
        auto &b1_pow = beta1_pow_t[neuron_index];
        auto &b2_pow = beta2_pow_t[neuron_index];

        if (m_weights.size() != static_cast<size_t>(input_size))
        {
            m_weights.assign(input_size, 0.0);
            v_weights.assign(input_size, 0.0);
            b1_pow = beta1;
            b2_pow = beta2;
            t = 1;
        }
        else
        {
            t += 1;
            b1_pow *= beta1;
            b2_pow *= beta2;
        }

        double correction1 = 1.0 / (1.0 - b1_pow);
        double correction2 = 1.0 / (1.0 - b2_pow);
        double lambda = (this->regularizer ? this->regularizer->lambda_value() : 0.0);

        for (int j = 0; j < input_size; ++j)
        {
            double grad = delta * input[j];
            if (lambda > 0.0)
                grad += lambda * weights[j];  // regularización L2

            m_weights[j] = beta1 * m_weights[j] + (1.0 - beta1) * grad;
            v_weights[j] = beta2 * v_weights[j] + (1.0 - beta2) * grad * grad;

            double m_hat = m_weights[j] * correction1;
            double v_hat = v_weights[j] * correction2;

            weights[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        double grad_b = delta;
        m_bias = beta1 * m_bias + (1.0 - beta1) * grad_b;
        v_bias = beta2 * v_bias + (1.0 - beta2) * grad_b * grad_b;

        double m_hat_b = m_bias * correction1;
        double v_hat_b = v_bias * correction2;

        bias -= learning_rate * m_hat_b / (std::sqrt(v_hat_b) + epsilon);
    }

    optimizer_type get_type() const override { return optimizer_type::ADAM; }
};
```

#### Variables y funciones `inline`

Se usan `inline` para definir variables constantes y funciones pequeñas, evitando múltiples definiciones y mejorando el rendimiento en compilación. Estas mappings permiten convertir entre el `enum` y cadenas legibles, facilitando el manejo de optimizadores por texto en configuraciones o logs.

```cpp
inline const std::unordered_map<optimizer_type, std::string> opt_to_string = {
    {optimizer_type::SGD, "sgd"},
    {optimizer_type::ADAM, "adam"},
    {optimizer_type::RMSPROP, "rmsprop"}};

inline const std::unordered_map<std::string, optimizer_type> string_to_opt = {
    {"sgd", optimizer_type::SGD},
    {"adam", optimizer_type::ADAM},
    {"rmsprop", optimizer_type::RMSPROP}};

inline std::string to_string(optimizer_type type)
{
    auto it = opt_to_string.find(type);
    return it != opt_to_string.end() ? it->second : "unknown";
}

inline optimizer_type from_string_opt(const std::string &str)
{
    auto it = string_to_opt.find(str);
    return it != string_to_opt.end() ? it->second : optimizer_type::SGD;
}
```

---

### Clase `DropoutController`

Implementa la técnica de **dropout**, un método de regularización que evita el sobreajuste en redes neuronales. Durante el entrenamiento, la clase desactiva aleatoriamente un subconjunto de neuronas (poniendo su activación en cero) con una probabilidad dada por `dropout_rate`.
Para mantener la escala de activaciones consistente, también incluye funciones para escalar las activaciones durante la inferencia (evaluación o prueba).

```cpp
class DropoutController
{
private:
    double dropout_rate;
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> dist;
}
```

#### `DropoutController(double rate)`

Constructor explícito que inicializa el controlador de dropout con un `rate` (valor entre 0 y 1), que indica la probabilidad con la que se "apagan" las neuronas durante el entrenamiento.

Internamente, también inicializa:

- Un generador de números aleatorios `rng` (con semilla aleatoria),
- Una distribución uniforme continua entre 0 y 1 para tomar decisiones de apagado.

```cpp
explicit DropoutController(double rate)
    : dropout_rate(rate), rng(std::random_device{}()), dist(0.0, 1.0) {}
```

#### `apply`

Aplica dropout **durante el entrenamiento**:

- Para cada valor de activación, genera un número aleatorio.
- Si el número es menor que `dropout_rate`, apaga (pone a cero) la neurona.
- Si no se apaga, **se escala la activación restante** por

![](.docs/f6.png)

para mantener la media constante.

Esto ayuda a prevenir que las neuronas dependan excesivamente unas de otras (co-adaptación).

```cpp
void apply(std::vector<double> &activations) const
{
    for (double &val : activations)
    {
        if (dist(rng) < dropout_rate)
            val = 0.0;
        else
            val *= (1.0 / (1.0 - dropout_rate));
    }
}
```

#### `scale_for_inference`

Escala todas las activaciones por $1 - \text{dropout\_rate}$ **durante la inferencia**, para compensar el hecho de que no se está aplicando apagado real.
Este paso es necesario si **no se escaló** durante el entrenamiento (es el caso inverso al anterior). Pero dado que tu `apply()` ya escala durante entrenamiento, este método es opcional y se puede omitir dependiendo del enfoque adoptado.

```cpp
void scale_for_inference(std::vector<double> &activations) const
{
    for (double &val : activations)
        val *= (1.0 - dropout_rate);
}
```

#### `rate()`

Retorna la tasa de dropout (`dropout_rate`) configurada en la clase. Útil si deseas consultarla desde otras clases o funciones de entrenamiento.

```cpp
double rate() const { return dropout_rate; }
```

---

### Clase `Regularizer`

La clase abstracta `Regularizer` define la interfaz para los regularizadores, mecanismos que penalizan los modelos complejos agregando un término al costo para evitar el sobreajuste.
Este diseño permite implementar distintas estrategias como **L1**, **L2** u otras variantes de regularización.

Permite dos operaciones clave:

- Calcular la penalización total dada una red (`compute_penalty`).
- Aplicar ajustes a los gradientes de los pesos (`apply`).

```cpp
class Regularizer
{
public:
    virtual double compute_penalty(const std::vector<Layer> &layers) const = 0;
    virtual void apply(std::vector<double> &weights, std::vector<double> &grad_weights) const = 0;
    virtual ~Regularizer() = default;
    virtual double lambda_value() const { return 0.0; }
};
```

#### Clase `L2Regularizer`

`L2Regularizer` implementa la regularización **L2** (también conocida como weight decay), que penaliza los pesos grandes sumando

![](.docs/f7.png)

al error.
Esto incentiva que los pesos se mantengan pequeños y distribuidos.

```cpp
class L2Regularizer : public Regularizer
{
private:
    double lambda;

public:
    explicit L2Regularizer(double lambda) : lambda(lambda) {}
    double compute_penalty(const std::vector<Layer> &layers) const override;
    void apply(std::vector<double> &weights, std::vector<double> &grad_weights) const override;
    double lambda_value() const;
};
```

##### `L2Regularizer`

Constructor que inicializa el regularizador con el valor lambda, que controla la fuerza de la penalización.

```cpp
explicit L2Regularizer(double lambda) : lambda(lambda) {}
```

##### `compute_penalty`

Itera sobre todas las capas y acumula la suma de los cuadrados de los pesos, delegando el cálculo parcial a `Layer::compute_penalty`.
Finalmente, devuelve

![](.docs/f8.png)

, como lo requiere la formulación estándar de L2.

```cpp
double L2Regularizer::compute_penalty(const std::vector<Layer> &layers) const
{
    double penalty = 0.0;
    for (const auto &layer : layers)
        layer.compute_penalty(penalty);
    return 0.5 * lambda * penalty;
}
```

##### `apply`

Ajusta cada gradiente de peso sumando

![](.docs/f9.png)

```cpp
void L2Regularizer::apply(std::vector<double> &weights, std::vector<double> &grad_weights) const
{
    for (size_t i = 0; i < weights.size(); ++i)
        grad_weights[i] += lambda * weights[i];
}
```

##### `lambda_value`

Devuelve el valor lambda actual del regularizador.

```cpp
double lambda_value() const { return lambda; }
```

---

### Ejecución

Con este comando se generará la carpeta `build` y dentro estarán los dos ejecutables `mlp_test` y `mlp_train`.

```bash
make run
```

#### Ejecución de Entrenamiento

##### **Modo A: Entrenar desde archivos `train.txt` y `test.txt`**

```bash
./build/mlp_train --dataset [path_dataset] --save_data [path_model.dat] --epochs [num_epochs] --config [path_config.txt]
```

- Ejemplo real:

```bash
./build/mlp_train --dataset ./database/MNIST/ --save_data ./output/MNIST/mnist_mlp.dat --epochs 5 --config ./config/mnist.txt
```

---

##### **Modo B: Generar datos desde MNIST binario y entrenar**

```bash
./build/mlp_train --mnist [train_samples] [test_samples] --epochs [num_epochs] --config [path_config.txt]
```

- Ejemplo real:

```bash
./build/mlp_train --mnist 20000 5000 --epochs 1 --config ./config/mnist.txt
```

##### Tabla de parámetros de `mlp_train`

| Parámetro              | Obligatorio                  | Descripción                                                                                                           |
| ---------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `--dataset [path]`     | ❌ (si usas `--mnist`)       | Ruta a la carpeta con `train.txt` y `test.txt`. Se usa también como destino si generas los datos desde MNIST binario. |
| `--save_data [file]`   | ✅                           | Ruta donde se guarda el modelo `.dat` entrenado o desde donde se carga uno existente.                                 |
| `--epochs [num]`       | ✅                           | Número de épocas para entrenar la red.                                                                                |
| `--config [file]`      | ✅                           | Archivo `.txt` con la configuración de la red neuronal.                                                               |
| `--mnist [train test]` | ❌ (si ya tienes los `.txt`) | Genera `train.txt` y `test.txt` desde los archivos binarios originales de MNIST.                                      |

---

#### Ejecución de Test o Consulta

##### Formato general del comando

```bash
./test_model --save_data [path_model.dat] --test [path_test.txt]
```

- Ejemplo real:

```bash
./test_model --save_data output/MNIST/mnist_mlp.dat --test dataset/mnist_style.txt
```

##### Tabla de parámetros

| Parámetro            | Obligatorio | Descripción                                                           |
| -------------------- | ----------- | --------------------------------------------------------------------- |
| `--save_data [file]` | ✅          | Ruta al archivo del modelo previamente entrenado (`.dat`).            |
| `--test [file]`      | ✅          | Ruta al archivo de prueba (`.txt`) con entradas y etiquetas en texto. |

### Salida

![Ejecución make](.docs/make.png)

#### Ejecución de Entrenamiento

![Ejecución TRAIN](.docs/train_mnits.png)

![Ejecución TRAIN](.docs/train_dataset.png)

#### Ejecución de Test

![Ejecución TEST](.docs/test.png)

## Implementación en Python

### test.py

Convierte un conjunto de imágenes PNG de dígitos (por ejemplo, del 0 al 9) y sus etiquetas correspondientes en un archivo de texto plano compatible con una red neuronal MLP. Lee las etiquetas desde un archivo (labels.txt) y las imágenes desde una carpeta, asegurándose de que la cantidad de imágenes coincida con la de etiquetas. Cada imagen se convierte a escala de grises, se redimensiona a 28×28 píxeles, se normaliza a valores entre 0 y 1 y se convierte en un vector de 784 características. Luego, cada etiqueta se transforma en codificación one-hot de 10 posiciones. El archivo de salida comienza con una línea "784 10" y contiene una línea por ejemplo con los valores de entrada seguidos por los de salida codificada. El resultado es un archivo de entrenamiento o prueba listo para usar con el programa MLP.

```python
import os
from PIL import Image
import numpy as np

def convert_dataset(image_folder, labels_file, output_file):
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith('.png')
    ])

    if len(image_files) != len(labels):
        print(f"Error: {len(image_files)} imágenes y {len(labels)} etiquetas.")
        return

    with open(output_file, 'w') as out:
        out.write("784 10\n")
        for i, filename in enumerate(image_files):
            path = os.path.join(image_folder, filename)
            img = Image.open(path).convert('L').resize((28, 28))
            data = np.array(img).astype(np.float32).flatten() / 255.0
            label = labels[i]
            one_hot = [0] * 10
            one_hot[label] = 1
            values = ' '.join(f"{x:.5f}" for x in data)
            label_str = ' '.join(str(v) for v in one_hot)
            out.write(f"{values} {label_str}\n")

    print(f"Dataset guardado en '{output_file}' con {len(image_files)} ejemplos.")

if __name__ == "__main__":
    image_folder = "./test/mnist45/"
    labels_file = "./test/mnist45/labels.txt"
    output_file = "./test/mnist45.txt"

    convert_dataset(image_folder, labels_file, output_file)

```

### graphic.py

Grafica los logs dedistinos maneras de entrenar el mnist para obtener distintos resultados, también comparandolos con un test externo.

```python
import matplotlib.pyplot as plt
import re
import os

def cargar_train_test_logs(path):
    epochs = []
    train_losses = []
    train_accs = []
    test_accs = []

    with open(path, 'r') as f:
        for line in f:
            match = re.match(
                r"Epoch\s+(\d+),\s+Train Loss:\s+([0-9.eE+-]+),\s+Train Acc:\s+([0-9.]+)%,\s+Train Time:.*?,\s+Test Acc:\s+([0-9.]+)%",
                line
            )
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)) / 100)
                test_accs.append(float(match.group(4)) / 100)
    return epochs, train_losses, train_accs, test_accs

logs = {
    "ADAM": "./output/MNIST_ADAM_001/log.txt",
    "RMS": "./output/MNIST_RMS_001/log.txt",
    "SGD": "./output/MNIST_SGD_001/log.txt"
}

global_epochs = None
global_test_accs = {}
global_train_accs = {}
global_train_losses = {}

for name, path in logs.items():
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {path}")
        continue

    epochs, train_losses, train_accs, test_accs = cargar_train_test_logs(path)
    global_epochs = epochs
    global_test_accs[name] = test_accs
    global_train_accs[name] = train_accs
    global_train_losses[name] = train_losses

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Entrenamiento y Evaluación - {name}")

    # Gráfico 1: Entrenamiento
    axs[0].plot(epochs, train_losses, label="Loss", color="red", linewidth=2)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss", color="red")
    axs[0].tick_params(axis='y', labelcolor="red")
    axs[0].set_title("Training Loss/Acc")
    axs[0].grid(True)

    ax2 = axs[0].twinx()
    ax2.plot(epochs, train_accs, label="Accuracy", color="blue", linestyle="--", linewidth=2)
    ax2.set_ylabel("Accuracy", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_ylim(0, 1.05)

    # Gráfico 2: Test Accuracy
    axs[1].plot(epochs, test_accs, label="Test Accuracy", color="green", linewidth=2)
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Test Accuracy")
    axs[1].set_title("Test Accuracy")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Gráfico TRAIN global
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

colors = {"SGD": "blue", "RMS": "red", "ADAM": "green"}

for name in global_train_losses:
    ax1.plot(global_epochs, global_train_losses[name], label=f"{name} Loss", linewidth=2, linestyle="-", color=colors[name])

for name in global_train_accs:
    ax2.plot(global_epochs, global_train_accs[name], label=f"{name} Accuracy", linewidth=2, linestyle="--", color=colors[name])

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Training Accuracy")
ax1.grid(True)
ax2.set_ylim(0, 1.05)

# Combinar ambas leyendas automáticamente y mostrarlas en el gráfico
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize="large")

plt.title("Comparación Global de Training Loss y Accuracy")
plt.tight_layout()
plt.show()



# Gráfico TEST global
plt.figure(figsize=(14, 8))
for name, accs in global_test_accs.items():
    plt.plot(global_epochs, accs, label=f"{name}", linewidth=2)

plt.title("Comparación Global de Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.ylim(0.95, 1.001)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


etiquetas = [
    "ADAM - 50 epochs",
    "ADAM - 40 epochs",
    "ADAM - 30 epochs",
    "ADAM - 20 epochs",
    "ADAM - 10 epochs",
    "RMS - 50 epochs",
    "RMS - 40 epochs",
    "RMS - 30 epochs",
    "RMS - 20 epochs",
    "RMS - 10 epochs",
    "SGD - 50 epochs",
    "SGD - 40 epochs",
    "SGD - 30 epochs",
    "SGD - 20 epochs",
    "SGD - 10 epochs",
]

test_acc_values = [0.35, 0.35, 0.35, 0.275, 0.35,
                   0.225, 0.275, 0.25, 0.325, 0.275,
                   0.35, 0.30, 0.35, 0.375, 0.425]
def extraer_epochs(label):
    return int(label.split('-')[1].strip().split()[0])
data = {
    "ADAM": [],
    "RMS": [],
    "SGD": []
}
epochs_ordenados = sorted({extraer_epochs(e) for e in etiquetas if e.startswith("ADAM")}, reverse=True)

for optim in data.keys():
    filtered = [(extraer_epochs(e), val) for e, val in zip(etiquetas, test_acc_values) if e.startswith(optim)]
    filtered.sort(key=lambda x: x[0], reverse=True)
    data[optim] = [val for _, val in filtered]

plt.figure(figsize=(12, 6))

colores = {"ADAM": "green", "RMS": "red", "SGD": "blue"}

for optim, valores in data.items():
    plt.plot(epochs_ordenados, valores, marker='o', linestyle='-', color=colores[optim], label=optim)

plt.xticks(epochs_ordenados, [f"{e} epochs" for e in epochs_ordenados], fontsize=10)
plt.ylim(0, 1.0)
plt.ylabel("External Test Accuracy")
plt.xlabel("Epochs")
plt.title("Comparación de Test Accuracy según configuración de entrenamiento")
plt.grid(True)
plt.legend(title="Optimizador")
plt.tight_layout()
plt.show()
```

### Ejecución

Para ejecutar el mlp_test, ejecutar esto antes.

```bash
python test.py
```

Y para comparar resultados ejecutar:

```bash
python graphic.py
```

### Salida

#### **Base**

- **SGD**: Entrenamiento base utilizando el optimizador SGD sin técnicas de regularización.

![](.docs/SGD.png)

- **RMSProp**: Entrenamiento base con el optimizador RMSProp sin dropout ni penalización L2.

![](.docs/RMS.png)

- **Adam**: Entrenamiento base con el optimizador Adam sin ningún tipo de regularización.

![](.docs/ADAM.png)

#### **Dropout**

- **Adam + Dropout 0.2**: Entrenamiento con Adam aplicando dropout con tasa de 0.2.

![](.docs/ADAM_DROP_02.png)

- **Adam + Dropout 0.5**: Entrenamiento con Adam aplicando dropout con tasa de 0.5.

![](.docs/ADAM_DROP_05.png)

#### **L2 (Weight Decay)**

- **Adam + L2 0.0001**: Entrenamiento con Adam utilizando regularización L2 con valor 0.0001.

![](.docs/ADAM_L2_0001.png)

- **Adam + L2 0.001**: Entrenamiento con Adam utilizando regularización L2 con valor 0.001.

![](.docs/ADAM_L2_001.png)

#### **Dropout + L2**

- **Adam + Dropout 0.5 + L2 0.001**: Entrenamiento combinado con Adam aplicando dropout con tasa 0.5 y regularización L2 con valor 0.001.

![](.docs/ADAM_DROP_05_L2_001.png)

- **Adam + Dropout 0.2 + L2 0.0001**: Entrenamiento combinado con Adam aplicando dropout con tasa 0.2 y regularización L2 con valor 0.0001.

![](.docs/ADAM_DROP_02_L2_0001.png)

#### **Evaluación en Test Externo (fuera de MNIST)**

- **Test Base**: Evaluación de la red entrenada sin regularización en un conjunto externo a MNIST.

![](.docs/TEST_BASE.png)

- **Test con Dropout**: Evaluación de la red entrenada con dropout (tasa 0.5 y 0.2) en un conjunto externo.

![](.docs/TEST_DROP.png)

- **Test con L2**: Evaluación de la red entrenada solo con regularización L2 (0.01 y 0.001) en un conjunto externo.

![](.docs/TEST_L2.png)

- **Test con Dropout + L2**: Evaluación de la red entrenada con dropout y regularización L2 en un conjunto externo.

![](.docs/TEST_DROP_L2.png)

## Tablas Comparativas

### Base

| Optimizador | Epoch | Train Acc (%) | Test Acc (%) | Comentario sobre Overfitting/Underfitting |
| ----------- | ----- | ------------- | ------------ | ----------------------------------------- |
| **Adam**    | 1     | 93.86         | 96.88        | Estable                                   |
|             | 6     | 98.28         | 97.08        | Leve overfitting empieza                  |
|             | 13    | 99.12         | 97.86        | Leve Overfitting                          |
|             | 20    | 99.47         | 97.85        | Leve overfitting continúa                 |
|             | 25    | 99.55         | 97.60        | Overfitting claro                         |
| **RMSProp** | 1     | 93.92         | 95.88        | Estable                                   |
|             | 6     | 98.11         | 97.19        | Overfitting                               |
|             | 13    | 98.84         | 97.47        | Overfitting                               |
|             | 20    | 99.29         | 97.55        | Overfitting visible                       |
|             | 25    | 99.36         | 97.23        | Overfitting claro                         |
| **SGD**     | 1     | 88.50         | 92.81        | Estable con posible Underfitting          |
|             | 6     | 96.76         | 96.60        | Overfitting                               |
|             | 13    | 98.34         | 97.54        | Overfitting                               |
|             | 20    | 99.00         | 97.78        | Overfitting                               |
|             | 25    | 99.32         | 97.73        | Muy claro overfitting                     |

### ADAM con DROPOUT

| Configuración          | Epoch | Train Acc (%) | Test Acc (%) | Comentario sobre Overfitting/Underfitting          |
| ---------------------- | ----- | ------------- | ------------ | -------------------------------------------------- |
| **Adam + Dropout 0.2** | 1     | 88.21         | 95.20        | Estable                                            |
|                        | 5     | 92.87         | 96.40        | Test mejora más rápido que el train                |
|                        | 13    | 93.32         | 96.66        | Buen ajuste                                        |
|                        | 17    | 93.33         | 96.59        | Estable, sin overfitting notable                   |
|                        | 24    | 93.19         | 95.77        | Test comienza a bajar, indicio de overfitting leve |
|                        | 25    | 93.43         | 96.19        | Cierre estable, ligera variación test              |
| **Adam + Dropout 0.5** | 1     | 88.26         | 94.91        | Estable                                            |
|                        | 5     | 92.99         | 96.76        | Muy buen test, generaliza mejor que el train       |
|                        | 11    | 93.73         | 96.61        | Test sigue alto, entrenamiento moderado            |
|                        | 19    | 93.70         | 96.58        | Estabilidad buena, sin signos de sobreajuste       |
|                        | 24    | 93.87         | 96.36        | Muy balanceado                                     |
|                        | 25    | 93.75         | 96.53        | Mejor estabilidad general que Dropout 0.2          |

### ADAM con L2

| Configuración       | Epoch | Train Acc (%) | Test Acc (%) | Comentario sobre Overfitting/Underfitting        |
| ------------------- | ----- | ------------- | ------------ | ------------------------------------------------ |
| **Adam + L2 0.01**  | 1     | 88.37         | 91.35        | Test ligeramente mejor que el train, buen inicio |
|                     | 5     | 90.31         | 91.78        | Curvas muy cercanas, sin signos de overfitting   |
|                     | 13    | 90.39         | 87.67        | **Overfitting** notorio (Test cae abruptamente)  |
|                     | 18    | 90.52         | 91.70        | Recuperación parcial del test                    |
|                     | 25    | 90.54         | 91.02        | Generalización limitada, indicios de sobreajuste |
| **Adam + L2 0.001** | 1     | 92.50         | 95.41        | **Overfitting** inicial leve (Test > Train)      |
|                     | 5     | 95.15         | 94.95        | Muy cercano, sin sobreajuste                     |
|                     | 13    | 95.22         | 95.37        | Buen equilibrio, generalización efectiva         |
|                     | 18    | 95.18         | 93.96        | **Overfitting leve**, test comienza a bajar      |
|                     | 25    | 95.11         | 95.24        | Estable, mejor generalización que con L2 0.01    |

### ADAM con DROPOUT + L2

| Configuración               | Epoch | Train Acc (%) | Test Acc (%) | Comentario sobre Overfitting/Underfitting                |
| --------------------------- | ----- | ------------- | ------------ | -------------------------------------------------------- |
| **Adam + Dropout + L2**     | 1     | 78.88         | 88.33        | Estable                                                  |
|                             | 5     | 79.37         | 86.62        | Test mejora más rápido, pero aún inestable               |
|                             | 10    | 79.17         | 88.36        | Test supera al train de forma sostenida                  |
|                             | 15    | 79.12         | 86.70        | Variabilidad entre epochs, sin mejora clara              |
|                             | 24    | 78.85         | 88.74        | **Underfitting leve**, Test sube pese a Train bajo       |
|                             | 25    | 78.84         | 86.36        | Dispersión final, modelo no converge bien                |
| **Adam + Dropout + L2 ALT** | 1     | 85.49         | 92.07        | Estable                                                  |
|                             | 5     | 87.06         | 93.04        | Mejora constante en ambos, generalización buena          |
|                             | 10    | 86.57         | 91.45        | Variación Test, posible ruido                            |
|                             | 15    | 86.71         | 93.68        | Buen ajuste, test alto y estable                         |
|                             | 24    | 86.21         | 93.73        | Test mejora incluso con caída de Train                   |
|                             | 25    | 86.04         | 93.82        | **Test alto sostenido**, ligera tendencia a overfitting. |

## Conclusiones

Las variaciones introducidas con técnicas de regularización como Dropout y L2 muestran impactos claros sobre el comportamiento de overfitting y underfitting en el entrenamiento con Adam. En la configuración base, Adam presenta un sobreajuste progresivo desde épocas tempranas: ya desde la época 6, la precisión de entrenamiento supera significativamente a la de prueba, manteniéndose así hasta el final del entrenamiento. Esto refleja un modelo que aprende rápidamente los patrones del conjunto de entrenamiento, pero que comienza a memorizar más de lo deseado, limitando su capacidad de generalización.

Cuando se introduce **Dropout**, tanto con probabilidad 0.2 como 0.5, el comportamiento del modelo cambia notablemente. Aunque las precisiones de entrenamiento disminuyen en comparación con la base, las de prueba se mantienen elevadas, y la brecha entre ambas se reduce considerablemente. Con Dropout 0.5, por ejemplo, el modelo logra una precisión de prueba superior al 96% de forma más estable, sin alcanzar valores extremos de entrenamiento. Esto indica un mejor balance, previniendo que el modelo se sobreentrene. En especial, el modelo con Dropout 0.5 muestra una curva de aprendizaje más controlada y sin signos de sobreajuste fuerte, lo cual sugiere que esta técnica ayuda a mantener la capacidad de generalización incluso al final del entrenamiento.

Por otro lado, **la regularización L2** genera resultados más variados. Con un valor alto como 0.01, se observa un claro **underfitting**, con precisiones de entrenamiento estancadas cerca del 90% y una caída pronunciada en la precisión de prueba en la época 13. En cambio, con un valor más bajo (0.001), el modelo muestra mejor equilibrio, alcanzando un ajuste adecuado sin sobreentrenarse, aunque reaparece un leve overfitting hacia las últimas épocas. La combinación de **Dropout y L2** tiende a frenar aún más el entrenamiento: en la configuración básica (Dropout + L2), el modelo no logra converger bien, manteniendo una precisión de entrenamiento muy baja (≈78%), mientras que la de prueba varía significativamente, indicando **underfitting**. Sin embargo, al ajustar los hiperparámetros (Dropout + L2 ALT), se consigue un mejor balance, con un test sostenidamente alto a pesar de un entrenamiento moderado, aunque hacia el final aparece una ligera tendencia al overfitting por la divergencia entre curvas. En conjunto, estos resultados evidencian cómo la elección y ajuste de técnicas de regularización impactan directamente en el equilibrio entre aprendizaje y generalización del modelo.

## Author

- **ShinjiMC** - [GitHub Profile](https://github.com/ShinjiMC)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
