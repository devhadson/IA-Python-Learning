## 2. Fórmulas Matemáticas del Teorema de Bayes (Naive Bayes)

Para el modelado probabilístico y el funcionamiento del clasificador se aplican tres conceptos fundamentales de la teoría de probabilidad, considerando el Teorema de Bayes:

* **Probabilidad Conjunta:** Mide la probabilidad de que dos eventos ocurran al mismo tiempo.

$$P(X = x, Y = y) = \frac{\text{Frecuencia de } (x, y)}{\text{Total de observaciones}}$$

* **Probabilidad Marginal:** La probabilidad de que ocurra un evento simple sin considerar otras variables. Se obtiene sumando las probabilidades conjuntas.

$$P(Y = y) = \sum_{x} P(X = x, Y = y)$$

* **Probabilidad Condicional:** La probabilidad de un evento dado que otro ya ha ocurrido.

$$P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}$$

* **Clasificador Naive Bayes (Inferencia):** Asume que las palabras clave ($X_1, X_2$) son condicionalmente independientes dado el estado del correo ($Y$). Para clasificar un correo que contiene ambas palabras, buscamos la clase que maximice el numerador del Teorema de Bayes:

$$P(Y \mid X_1, X_2) \propto P(Y) \cdot P(X_1 \mid Y) \cdot P(X_2 \mid Y)$$

Donde la predicción final se define mediante el argumento que maximiza la probabilidad (Maximum A Posteriori):

$$\hat{Y} = \arg\max_{y \in \{0,1\}} P(Y = y) \prod_{i=1}^{n} P(X_i = x_i \mid Y = y)$$