import requests
import numpy as np

# Archivo que queremos descargar (normalmente solo tendremos que reemplazar lo que esta luego de la ultima '/'
respuesta = requests.get("https://raw.githubusercontent.com/lbiedma/an2famaf2020/master/datos/dryer2.dat")

# Leer el arreglo (unidimensional) que viene de los datos en forma de texto con separador TAB en este caso
A = np.fromstring(respuesta.text, sep="\t")

# Darle forma al arreglo, en este caso sabemos que tenemos 7 columnas

A = A.reshape(int(len(A) / 7), 7)

