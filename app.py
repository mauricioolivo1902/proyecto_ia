
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.title('Prediccion de Notas según Horas de Estudio')
st.write('Este modelo de Regresión Lineal Simple predice tu calificación basada en el tiempo de estudio.')

# 1. Cargar los datos
df = pd.read_csv('datos_regresion.csv')
X = df[['Horas_Estudio']]
y = df['Nota']

# 2. Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 3. Interfaz de usuario (Barra lateral para predicción)
st.sidebar.header('¡Haz tu predicción!')
horas_input = st.sidebar.slider('Selecciona cuántas horas estudias:', 0.0, 10.0, 5.0)

# Realizar predicción
prediccion = modelo.predict([[horas_input]])
st.sidebar.subheader(f'Nota estimada: {prediccion[0]:.2f}')

# 4. Gráfica interactiva
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', alpha=0.5, label='Datos históricos')
ax.plot(X, modelo.predict(X), color='red', label='Línea de tendencia')
ax.scatter(horas_input, prediccion, color='green', s=200, marker='*', label='Tu predicción') # Punto del usuario
ax.set_xlabel('Horas de Estudio')
ax.set_ylabel('Nota')
ax.legend()

st.pyplot(fig)

st.write("### Cómo funciona:")
st.write(f"La ecuación del modelo es: **Nota = {modelo.coef_[0]:.2f} * Horas + {modelo.intercept_:.2f}**")
