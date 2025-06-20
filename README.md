# Convolutional Neural Network - Convolutional

By Braulio Nayap Maldonado Casilla

## Introducción

La convolución es una operación matemática fundamental en el procesamiento de imágenes y en las redes neuronales convolucionales (CNN, por sus siglas en inglés). Esta operación permite extraer patrones locales de una entrada multidimensional, como bordes, texturas o estructuras complejas, mediante el uso de filtros o kernels que se deslizan sobre la imagen de entrada.

En el contexto de las CNN, una convolución consiste en aplicar un conjunto de filtros sobre una entrada (típicamente una imagen con varios canales), generando como resultado una nueva representación llamada mapa de características (feature map). Cada filtro es un pequeño tensor de pesos entrenables que se ajustan durante el aprendizaje para detectar patrones específicos.

### Fórmula de la convolución discreta 2D

La salida en la posición (i, j) del canal de salida c_out, aplicando una convolución sobre una entrada multicanal, se calcula como:

![](.docs/f1.png)

| Símbolo              | Significado                                                  |
| -------------------- | ------------------------------------------------------------ |
| Y_c_out (i, j)       | Valor en la posición (i,j) del canal de salida c_out         |
| X_c_in (i,j)         | Valor en la posición (i,j) del canal de entrada c_in         |
| K_c_out, c_in (m, n) | Valor del kernel que conecta c_in => c_out en posición (m,n) |
| b_c_out              | Sesgo (bias) correspondiente al canal de salida c_out        |
| S                    | Stride (paso de desplazamiento)                              |
| P                    | Padding (relleno con ceros alrededor de la entrada)          |
| C_in                 | Número de canales de entrada                                 |
| C_out                | Número de filtros / canales de salida                        |
| K_H, K_W             | Altura y ancho del kernel                                    |

### Fórmulas para calcular el tamaño de salida de la convolución

Dada una entrada tridimensional de tamaño:

![](.docs/f2.png)

y una convolución definida por:

- Kernel de tamaño K_H x K_W
- Stride S
- Padding P
- Número de filtros C_out

El tamaño de la salida será:

![](.docs/f3.png)

| Símbolo | Significado                                       |
| ------- | ------------------------------------------------- |
| C_in    | Número de canales de la entrada (ej. 3 si es RGB) |
| H_in    | Altura (filas) de la entrada                      |
| W_in    | Ancho (columnas) de la entrada                    |
| K_H     | Altura del kernel (filtro)                        |
| K_W     | Ancho del kernel (filtro)                         |
| P       | Padding (número de ceros añadidos en los bordes)  |
| S       | Stride (paso con el que se desplaza el filtro)    |
| C_out   | Número de filtros (canales de salida)             |
| H_out   | Altura del mapa de características de salida      |
| W_out   | Ancho del mapa de características de salida       |
| ⌊⌋      | Parte entera inferior (redondeo hacia abajo)      |

---

## Implementación en C++

---

## Salida

![](.docs/original.png)

![](.docs/channel0.png)

![](.docs/channel1.png)

---

## Conclusiones

Las variaciones introducidas con técnicas de regularización como Dropout y L2 muestran impactos claros sobre el comportamiento de overfitting y underfitting en el entrenamiento con Adam. En la configuración base, Adam presenta un sobreajuste progresivo desde épocas tempranas: ya desde la época 6, la precisión de entrenamiento supera significativamente a la de prueba, manteniéndose así hasta el final del entrenamiento. Esto refleja un modelo que aprende rápidamente los patrones del conjunto de entrenamiento, pero que comienza a memorizar más de lo deseado, limitando su capacidad de generalización.

Cuando se introduce **Dropout**, tanto con probabilidad 0.2 como 0.5, el comportamiento del modelo cambia notablemente. Aunque las precisiones de entrenamiento disminuyen en comparación con la base, las de prueba se mantienen elevadas, y la brecha entre ambas se reduce considerablemente. Con Dropout 0.5, por ejemplo, el modelo logra una precisión de prueba superior al 96% de forma más estable, sin alcanzar valores extremos de entrenamiento. Esto indica un mejor balance, previniendo que el modelo se sobreentrene. En especial, el modelo con Dropout 0.5 muestra una curva de aprendizaje más controlada y sin signos de sobreajuste fuerte, lo cual sugiere que esta técnica ayuda a mantener la capacidad de generalización incluso al final del entrenamiento.

Por otro lado, **la regularización L2** genera resultados más variados. Con un valor alto como 0.01, se observa un claro **underfitting**, con precisiones de entrenamiento estancadas cerca del 90% y una caída pronunciada en la precisión de prueba en la época 13. En cambio, con un valor más bajo (0.001), el modelo muestra mejor equilibrio, alcanzando un ajuste adecuado sin sobreentrenarse, aunque reaparece un leve overfitting hacia las últimas épocas. La combinación de **Dropout y L2** tiende a frenar aún más el entrenamiento: en la configuración básica (Dropout + L2), el modelo no logra converger bien, manteniendo una precisión de entrenamiento muy baja (≈78%), mientras que la de prueba varía significativamente, indicando **underfitting**. Sin embargo, al ajustar los hiperparámetros (Dropout + L2 ALT), se consigue un mejor balance, con un test sostenidamente alto a pesar de un entrenamiento moderado, aunque hacia el final aparece una ligera tendencia al overfitting por la divergencia entre curvas. En conjunto, estos resultados evidencian cómo la elección y ajuste de técnicas de regularización impactan directamente en el equilibrio entre aprendizaje y generalización del modelo.

## Author

- **ShinjiMC** - [GitHub Profile](https://github.com/ShinjiMC)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
