# Vamos a resolver los 3 incisos en diferentes funciones.
# Para correr esto se puede hacer from p3ej14 import ejercicio_* y ejecutar ejercicio_*(),
# donde * es a o b.
import numpy as np
import matplotlib.pyplot as plt


def ejercicio_a():
    # Hay que armar la lista de epsilon que vamos a visitar.
    # Para que tienda a 0 podemos usar algo como 1/n
    ns = np.arange(1, 100)
    epsilon = 1.0 / (ns)

    # Tenemos una lista que vamos a empezar a poblar con dets y Ks
    dets = []
    ks = []
    # Determinante de esta matriz es siempre 1
    det = 1.0
    # Hacemos un loop
    for eps in epsilon:
        A = np.array([[1.0, 1.0 - eps], [0.0, 1.0]])
        dets.append(det)
        # Numero de condicion viene en la sublibrería linalg
        k = np.linalg.cond(A)
        ks.append(k)

    # Debemos plotear los pasos de la sucesión, así que avanzamos en N
    plt.plot(ns, dets, '*r', ns, ks, '.b')
    plt.show()


def ejercicio_b():
    # Hay que armar la lista de epsilon que vamos a visitar.
    # Para que tienda a 0 podemos usar algo como 1/n^2
    ns = np.arange(1, 100)
    epsilon = 1.0 / (ns * ns)

    # Tenemos una lista que vamos a empezar a poblar con dets y Ks
    dets = []
    ks = []
    # Determinante de esta matriz es siempre 1
    det = 1.0
    # Hacemos un loop
    for eps in epsilon:
        A = np.array([[1.0 / eps, 0.0], [0.0, 1.0]])
        dets.append(det)
        # Numero de condicion viene en la sublibrería linalg
        k = np.linalg.cond(A)
        ks.append(k)

    # Debemos plotear los pasos de la sucesión, así que avanzamos en N
    plt.plot(ns, dets, '*r', ns, ks, '.b')
    plt.show()


# La ejecución de esta función es diferente, el input debe ser un epsilon.
def ejercicio_c(eps):
    # Armamos nuestra lista de vectores de la esfera unidad ayudándonos de
    # nuestras queridas funciones trigonométricas y 100 puntos.

    intervalo = np.linspace(0, 2 * np.pi, 100)
    # Vamos a poner a todos nuestros puntos en una matriz 2x100, donde la primera
    # fila es la de los x y la segunda de los y.
    puntos = np.array([np.cos(intervalo), np.sin(intervalo)])
    # A es la primera matriz
    A = np.array([[1.0, 1.0 - eps], [0.0, 1.0]])
    # Los transformamos a todos al mismo tiempo con la ayuda del broadcasting.
    puntos_primera_trans = A @ puntos

    # B es la segunda matriz
    B = np.array([[1.0 / eps, 0.0], [0.0, 1.0]])
    # Transformamos de nuevo
    puntos_segunda_trans = B @ puntos

    # Hacemos el plot en 3 lineas diferentes para mejor organización
    # Recordar que vamos a tener 3 matrices 2x100, entonces debemos graficar
    # primera fila (los x) vs. segunda fila (los y)
    # Con label le damos la etiqueta a los datos
    plt.plot(puntos[0, :], puntos[1, :], label="Esfera Original")
    plt.plot(puntos_primera_trans[0, :], puntos_primera_trans[1, :], label="Primera transformación")
    plt.plot(puntos_segunda_trans[0, :], puntos_segunda_trans[1, :], label="Segunda transformación")
    # Con legend mostramos las etiquetas en el plot
    plt.legend()
    # Con 'equal' pedimos que las escalas sean iguales en x e y.
    plt.axis('equal')

    plt.show()
