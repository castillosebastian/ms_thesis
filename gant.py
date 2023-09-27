import matplotlib.pyplot as plt
import numpy as np

# Define the activities and their durations in weeks
activities = [
    "Revisión de Literatura",
    "Configuración del Entorno en VSCode",
    "Selección de Datasets y Preprocesamiento de Datos",
    "Implementación del Algoritmo Genético clásico",
    "Pruebas Preliminares del Algoritmo Genético",
    "Estudio de Autoencoders Variacionales",
    "Implementación de Autoencoder Variacional",
    "Integración del Autoencoder Variacional con AG",
    "Pruebas de Integración entre AG y AV",
    "Definición y configuración de Métricas de Desempeño",
    "Implementación de la Evaluación",
    "Ejecución de Pruebas de Desempeño",
    "Análisis de Resultados",
    "Selección e implementación de Métodos del Estado del Arte",
    "Análisis Comparativo",
    "Documentación del Proyecto",
    "Redacción de la Tesis",
    "Revisión y Corrección de la Tesis"
]

durations = [3, 3, 3, 6, 2, 4, 3, 4, 2, 2, 4, 3, 4, 3, 3, 3, 3, 3]

# Calculate cumulative sum to get the ending week for each task
cumulative_durations = np.cumsum(durations)

# Invert the order of activities for the Gantt chart
reversed_activities = activities[::-1]
reversed_cumulative_durations = np.cumsum(durations[::-1])

# Correct the mistake by reversing both the activities and durations
# This time, let's keep track of the starting week for each activity to ensure the data aligns correctly

# Initialize the starting week
start_week = 1

# Create the corrected Gantt chart using Matplotlib
fig, ax = plt.subplots(figsize=(12, 10))

# Loop through activities and durations in reversed order
for i, (activity, duration) in enumerate(zip(reversed(activities), reversed(durations))):
    end_week = start_week + duration - 1  # Calculate the ending week for the current activity
    ax.broken_barh([(start_week, duration)], (i-0.4, 0.8), facecolors='blue')
    ax.text(start_week + duration/4, i, f"{start_week}-{end_week}", color="white", fontsize=10)
    start_week = end_week + 1  # Update the starting week for the next activity

# Configure the axes and labels
ax.set_xlabel("Weeks")
ax.set_yticks(range(len(activities)))
ax.set_yticklabels(reversed(activities))
ax.grid(True)

plt.title("Corrected Gantt Chart for Master's Thesis Project")
plt.show()
