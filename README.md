# MLP (Multi-Layer Perceptron)

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

---

Claro, aquí tienes una **explicación completa de la clase `Mlp`** siguiendo el estilo del template anterior:

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

public:
    Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
        double lr, std::vector<ActivationType> activation_types);
    Mlp() = default;
    void forward(const std::vector<double> &input, std::vector<std::vector<double>> &activations);
    void backward(const std::vector<double> &input,
                  const std::vector<std::vector<double>> &activations,
                  const std::vector<double> &expected);
    void one_hot_encode(int label, std::vector<double> &target);
    double cross_entropy_loss(const std::vector<double> &predicted, const std::vector<double> &expected);
    void train(std::vector<std::vector<double>> &images, std::vector<int> &labels,
               double &average_loss, double &train_accuracy);
    void test(const std::vector<std::vector<double>> &images, const std::vector<int> &labels, double &test_accuracy);
    void train_test(std::vector<std::vector<double>> &train_images, std::vector<int> &train_labels,
                    const std::vector<std::vector<double>> &test_images, const std::vector<int> &test_labels,
                    bool Test, const std::string &dataset_filename, int epochs = 1000);
    void save_data(const std::string &filename) const;
    bool load_data(const std::string &filename);
    void test_info(const std::vector<std::vector<double>> &X_test, const std::vector<int> &y_test);
};
```

#### Constructor

Este constructor crea e inicializa una red neuronal multicapa (MLP) configurando su arquitectura y parámetros principales. Recibe el número de entradas (`n_inputs`), una lista con el tamaño de cada capa oculta (`layer_sizes`), el número de salidas (`n_outputs`), la tasa de aprendizaje (`lr`), y las funciones de activación para cada capa (`activation_types`). Primero verifica que la cantidad de capas coincida con la cantidad de funciones de activación. Luego, recorre cada capa, construyendo objetos `Layer` con el tamaño correspondiente (entradas y salidas) y su función de activación asociada. Este proceso establece la estructura completa de la red, desde las entradas hasta la última capa, lista para el entrenamiento.

```cpp
Mlp(int n_inputs, const std::vector<int> &layer_sizes, int n_outputs,
         double lr, std::vector<ActivationType> activation_types)
    : n_inputs(n_inputs), n_outputs(n_outputs), learning_rate(lr)
{
    if (layer_sizes.size() != activation_types.size())
    {
        throw std::invalid_argument("layer_sizes and activation_types must have the same length.");
    }
    int prev_size = n_inputs;
    for (size_t i = 0; i < layer_sizes.size(); ++i)
    {
        layers.emplace_back(prev_size, layer_sizes[i], activation_types[i]);
        prev_size = layer_sizes[i];
    }
}
```

#### `forward`

La función `forward` implementa la propagación hacia adelante en la red neuronal, calculando las salidas capa por capa a partir de una entrada dada. Comienza limpiando el vector de activaciones y almacenando la entrada original como la primera activación. Luego, para cada capa en la red, se crea un vector de salida del tamaño adecuado y se calcula la salida de la capa usando la función `linear_forward`, que aplica los pesos, el sesgo y la función de activación correspondiente. Cada resultado se añade a la lista de activaciones, permitiendo conservar el estado completo de la red en cada paso, lo cual es fundamental para el posterior proceso de retropropagación (`backward`).

```cpp
void forward(const std::vector<double> &input, std::vector<std::vector<double>> &activations)
{
    activations.clear();
    activations.push_back(input);

    for (auto &layer : layers)
    {
        std::vector<double> output(layer.get_output_size());
        layer.linear_forward(activations.back(), output);
        activations.push_back(output);
    }
}
```

#### `backward`

Implementa la retropropagación del error desde la capa de salida hacia las capas anteriores, calculando los deltas (gradientes) para cada neurona según la diferencia entre la salida actual y la esperada, ajustada por la derivada de la función de activación. Luego, utiliza estos deltas para actualizar los pesos y sesgos de cada neurona, aplicando la regla del gradiente descendente con la tasa de aprendizaje, con el fin de minimizar el error de la red.

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

    // Actualización de pesos y biases
    for (size_t l = 0; l < layers.size(); ++l)
    {
        for (int i = 0; i < layers[l].get_output_size(); ++i)
        {
            for (int j = 0; j < layers[l].get_input_size(); ++j)
                layers[l].get_neurons()[i].get_weights()[j] -= learning_rate * deltas[l][i] * activations[l][j];
            layers[l].get_neurons()[i].get_bias() -= learning_rate * deltas[l][i];
        }
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

Entrena la red usando un conjunto de imágenes y etiquetas realizando un pase completo para cada muestra: primero convierte la etiqueta a vector one-hot, luego ejecuta la propagación hacia adelante para obtener la predicción, calcula la pérdida con entropía cruzada y aplica la retropropagación para ajustar los pesos. Además, acumula la pérdida total y cuenta las predicciones correctas para actualizar la pérdida promedio y la precisión de entrenamiento al finalizar el ciclo.

```cpp
void train(std::vector<std::vector<double>> &images, std::vector<int> &labels,
        double &average_loss, double &train_accuracy)
{
    size_t n = images.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    std::vector<double> target(n_outputs);
    std::vector<std::vector<double>> activations;
    double total_loss = 0.0;
    int correct = 0;

    for (size_t k = 0; k < n; ++k)
    {
        size_t i = indices[k];
        const auto &input = images[i];
        int label = labels[i];
        one_hot_encode(label, target);
        forward(input, activations);

        total_loss += cross_entropy_loss(activations.back(), target);
        backward(input, activations, target);

        int pred = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == label)
            ++correct;
    }
    average_loss = total_loss / n;
    train_accuracy = 100.0 * correct / n;
}
```

#### `test`

Evalúa la red sobre un conjunto de datos de prueba sin modificar los pesos, realizando una propagación hacia adelante para cada muestra y calculando la precisión como el porcentaje de predicciones correctas respecto a las etiquetas reales.

```cpp
void test(const std::vector<std::vector<double>> &images, const std::vector<int> &labels, double &test_accuracy)
{
    int correct = 0;
    std::vector<std::vector<double>> activations;

    for (size_t i = 0; i < images.size(); ++i)
    {
        forward(images[i], activations);
        int pred = std::distance(activations.back().begin(), std::max_element(activations.back().begin(), activations.back().end()));
        if (pred == labels[i])
            ++correct;
    }

    test_accuracy = 100.0 * correct / images.size();
}
```

#### `train_test`

Gestiona el ciclo completo de entrenamiento y evaluación de la red durante un número determinado de épocas, ejecutando en cada época el entrenamiento con el conjunto de datos de entrenamiento y, opcionalmente, la evaluación con el conjunto de prueba para medir la precisión. Además, registra el progreso en un archivo de log, muestra estadísticas por consola, guarda el modelo periódicamente y detiene el proceso anticipadamente si se cumplen criterios de convergencia o precisión deseada.

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
            log_line << ", Test Acc: " << test_accuracy << "%"
                     << ", Test Time: " << test_time << "s";

        std::cout << log_line.str() << std::endl;
        log_file << log_line.str() << std::endl;

        if (average_loss < 0.001 || test_accuracy > 98.0 || epoch >= epochs)
        {
            std::cout << "Stopping training: early stopping criteria met.\n";
            break;
        }
        if ((epoch + 1) % 10 == 0)
        {
            std::string filename = (output_dir / ("epoch_" + std::to_string(epoch + 1) + ".dat")).string();
            save_data(filename);
            std::cout << "Model saved at epoch " << (epoch + 1) << " to " << filename << ".\n";
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

public:
    bool load_config(const std::string &filename, const int inputs);
    const std::vector<int> &get_layer_sizes() const;
    float get_learning_rate() const;
    const std::vector<ActivationType> &get_activations() const;
    const void print_config();
};
```

#### Método `load_config`

Carga los parámetros desde un archivo de texto siguiendo este formato:

```
[capas]          // línea 1: tamaños de capa separados por espacio (ej: 32 16 10)
[lr]             // línea 2: tasa de aprendizaje (ej: 0.01)
[activaciones]   // línea 3: nombres de funciones por capa (ej: relu relu softmax)
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
    std::cout << "==========================\n";
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

![Ejecución TRAIN](.docs/train.png)

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
graphic.py
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

![Grafico](.docs/graphics.png)

## Conclusiones

Los resultados del entrenamiento mostraron que, para problemas

## Author

- **ShinjiMC** - [GitHub Profile](https://github.com/ShinjiMC)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
