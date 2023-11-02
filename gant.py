import matplotlib.pyplot as plt
import numpy as np

# Updated activities and durations
activities = [
    #"Revisión de Literatura",
    #"Configuración del Entorno en VSCode",
    "Seleccionar y Preprocesar Datos",
    "Implementar Algoritmo Genético (AG)",
    #"Pruebas Preliminares del Algoritmo Genético",
    #"Estudio de Autoencoders Variacionales",
    "Implementar Autoencoder Variacional (AV)",
    "Integrar AV y AG",
    #"Pruebas de Integración entre AG y AV",
    #"Evaluar Desempeño",
    #"Implementación de la Evaluación",
    #"Ejecución de Pruebas de Desempeño",
    "Realizar Experimentos y Analizar Resultados",
    "Comparar con Otros Métodos ",
    "Analizar Resultados",
    #"Documentación del Proyecto",
    "Redacción de la Tesis",
    #"Revisión y Corrección de la Tesis"
]
durations = [4,4,4,5,7,4,4,6]

# Create an 'n' array to enumerate the activities
n = list(range(1, len(activities) + 1))
n = n[::-1]
# Initialize the starting week
start_week = 1

# Create the updated Gantt chart with a new 'n' column
fig, ax = plt.subplots(figsize=(14, 10))
fig.subplots_adjust(left=0.4)  # Adjust the left margin

# Loop through activities, durations, and n in regular (not reversed) order
for i, (activity, duration, num) in enumerate(zip(activities, durations, n)):
    end_week = start_week + duration - 1  # Calculate the ending week for the current activity
    ax.broken_barh([(start_week, duration)], ((len(activities) - i - 1) - 0.4, 0.8), facecolors='blue')
    ax.text(end_week + 1, (len(activities) - i - 1), f"{start_week}-{end_week}", color="black", fontsize=12)
    start_week = end_week + 1  # Update the starting week for the next activity

# Configure the axes and labels
ax.set_xlabel("semanas", fontsize=14)
ax.set_yticks(range(len(activities)))
ax.set_yticklabels([f"{num}. {activity}" for num, activity in zip(n, reversed(activities))], fontsize=14)
ax.grid(True)

plt.title("Gantt - Plan de Trabajo Tesis", fontsize=16)
# Save the Gantt chart as a PNG file
save_path = 'data/gantt_tesis.png'
plt.savefig(save_path)