import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#leer los datos de entrenamiento
temperaturas = pd.read_csv("C:/Users/David/Downloads", sep=",")

#separar los datos de entrada y salida
celsius = temperaturas["celsius"]
fahrenheit = temperaturas['fahrenheit']

modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

modelo.compile(optimizer=tf.keras.optimizers.Adam(1),loss='mean_squared_error')

epocas = modelo.fit(celsius, fahrenheit, epochs=100, verbose=0)

f = modelo.predict([27])
print(f)