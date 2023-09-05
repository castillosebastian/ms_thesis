UTN-Maestría en Minería de Datos
Plan de Tesis

# Aplicación de autoencoders variacionales para mejorar los procesos de optmización evolutiva multiobjetivo

Alumno: Claudio Sebastián Castillo   
Directores: Dr. Matías Gerard y Dr. Leandro Vignolo


## Introducción

La optimización es un componente central del diseño de los algoritmos de aprendizaje automático, contribuyendo significativamente a su rendimiento y extendida aplicación. En contextos que involucran funciones objetivo de naturaleza simple y diferenciable, los problemas de optimización pueden abordarse de manera eficiente a través de métodos determinísticos, como por ejemplo el cálculos de gradientes. Sin embargo en problemas de optimización multimodales, altamente no lineales y complejos -i.e. un basto campo de situaciones reales- las técnicas determinísticas resultan ineficaces, planteando la necesidad de buscar métodos diferentes.

Los algoritmos evolutivos (en adelante AE) son métodos de optimización inspirados en la evolución natural, diseñados para encontrar soluciones en espacios de búsqueda complejos (Smith, 2015). A diferencia de los métodos de optimización determinísticos, los algoritmos evolutivos son particularmente efectivos en espacios de búsqueda discretos o cuando la función objetivo es desconocida o no diferenciable (Williams, 2018). Utilizando técnicas como la selección, la mutación y el cruce, estos algoritmos generan iterativamente nuevas soluciones a partir de una población de candidatos (Jones et al., 2020), de manera similar a cómo la evolución natural optimiza las características biológicas a lo largo del tiempo (Smith, 2015).

Pero como sucede con gran parte de los algoritmos de aprendizaje automático, los AE también se enfrentan a problemas desafiantes cuando se aplican a datasets de alta dimensionalidad y bajo número de muestras. En el contexto de espacios con dimensionalidad elevada, la cardinalidad del conjunto de soluciones candidatas se incrementa de manera exponencial, lo cual puede conducir a un problema computacionalmente intratable **citar 2n**. Simultáneamente, la escasa disponibilidad de datos representa un obstáculo en la optimización de la función objetivo, ya que limita la capacidad informativa de la misma y, por ende, incrementa la probabilidad de converger hacia soluciones subóptimas (Hastie et al., 2009).

En este contexto, la aumentación de datos mediante Autoencoders Variacionales se presenta como una estrategia prometedora al problema de la escases de datos. Estos modelos generativos son capaces de aprender una representación latente de los datos de entrada y generar nuevos datos que mantienen las mismas características fundamentales (i.e.similar distribución conjunta de probabilidad)  que los datos originales. Al expandir el conjunto de datos de esta manera, se mejora la robustez del proceso evolutivo, permitiendo una exploración más efectiva del espacio de soluciones y mitigando los riesgos de sobreajuste y convergencia prematura. De este modo, la combinación de algoritmos evolutivos con técnicas de aumentación de datos puede ofrecer un enfoque eficaz para abordar problemas de optimización en escenarios de alta dimensionalidad y muestras escasas.

## Definición del problema

La escasez de datos en procesos de optimización evolutiva multiobjetivo y su posible solución a partir de técnicas de aumentación mediante autoencoders variacionales.

## Fundamentación y justificación del tema (extensión máxima 2 páginas)
• Marco teórico.
• Valor científico del trabajo propuesto.
• Alcance.

## Estado del arte (extensión máxima sugerida 2 páginas)
• Evolución histórica y actual del conocimiento.
• Aspectos o conocimiento que se encuentre vacantes.




