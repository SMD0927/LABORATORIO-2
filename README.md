# Convolución, correlación y transformación 
 LABORATORIO - 2 PROCESAMIENTO DIGITAL DE SEÑALES


## Requisitos
- Python 3.9
- Bibliotecas necesarias:
  - wfdb
  - numpy
  - matplotlib
  - seaborn

Instalar dependencias:
pip install wfdb numpy matplotlib seaborn
##Introducción
Este proyecto presenta ejemplos prácticos de procesamiento digital de señales mediante operaciones fundamentales como la convolución, la correlación y la transformada de Fourier, además del análisis de una señal electromiográfica (EMG) en funcion de la frecuencia.

-----

## Convolución
La convolución es una operación que permite obtener la respuesta de un sistema LTI (lineal e invariante en el tiempo) a partir de su respuesta al impulso y de una señal de entrada. La salida se obtiene como la suma ponderada de la señal de entrada desplazada y multiplicada por la respuesta del sistema.
### Fórmula de la convolución discreta:

$$
y[n] = \sum_{k=0}^{M-1} x[k] h[n-k]
$$

Donde:

- \(y[n]\) es la señal de salida.
- \(x[k]\) es la señal de entrada.
- \(h[n-k]\) es la respuesta al impulso del sistema desplazada en el tiempo.
- \(M\) es la longitud de la señal de entrada.

### 1. Convolución entre la señal x[n] y del sistema h[n]
```python
h = [5,6,0,0,7,7,5]
x = [1,0,1,4,6,6,0,7,0,8]
y = np.convolve(x,h,mode='full')
print('h[n] =', h)
print('x[n] =',x)
print('y[n] =',y)
```
$$
h[n] = \begin{bmatrix}
5 & 6 & 0 & 0 & 7 & 7 & 5
\end{bmatrix}
$$

$$
x[n] = \begin{bmatrix}
1 & 0 & 1 & 4 & 6 & 6 & 0 & 7 & 0 & 8
\end{bmatrix}
$$

$$
y[n] = \begin{bmatrix}
5 & 6 & 5 & 26 & 61 & 73 & 48 & 70 & 117 & 144 & 120 & 79 & 49 & 91 & 56 & 40
\end{bmatrix}
$$

Este código en Python calcula la convolución discreta entre dos señales utilizando la función np.convolve() de NumPy. Primero, se definen dos listas, h y x, que representan la respuesta al impulso de un sistema y una señal de entrada, respectivamente. Luego, se aplica la convolución entre estas dos señales usando np.convolve(x, h, mode='full'), lo que genera una nueva señal y cuya longitud es la suma de las longitudes de x y h menos uno. La convolución es una operación fundamental en procesamiento de señales, ya que permite analizar cómo una señal se ve afectada por un sistema. Finalmente, el código imprime las señales h, x y y para visualizar los datos y el resultado de la convolución.

---

### 2. Grafico de la señal x[n] y del sistema h[n]
```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(h,color='g')
plt.stem(range(len(h)), h)
plt.title("Sistema (santiago)")  
plt.xlabel("(n)") 
plt.ylabel("h [n]") 
plt.grid()
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/b400a6c8-f58f-4757-a74c-ed36a19d3d59" alt="imagen" width="450">
</p>

```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(x,color='g')
plt.stem(range(len(x)), x)
plt.title("Señal (santiago)")  
plt.xlabel("(n)") 
plt.ylabel("x [n]") 
plt.grid()  
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/bfa4f9b0-51ed-40c3-b052-ca6c8a513123" alt="imagen" width="450">
</p>

Este código genera dos gráficos para representar la respuesta al impulso h[n] y la señal de entrada x[n]. Para cada una, se crea una figura de 10x5 y se trazan dos representaciones: una línea verde (plt.plot()) y un gráfico de tipo stem (plt.stem()) para resaltar los valores discretos.

---

### 3. Grafico de la convolución
```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(y,color='g')
plt.title("Señal Resultante (santiago)")  
plt.xlabel("(n)") 
plt.ylabel("y [n]") 
plt.grid() 
plt.stem(range(len(y)), y)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/df85b514-81c1-4ea5-bc03-17c59fa7ca0d" alt="imagen" width="450">
</p>

Este fragmento de código genera un gráfico de la señal resultante y[n], que es el resultado de la convolución entre x[n] y h[n]. Se traza la señal con una línea verde usando plt.plot(y, color='g'). Luego, se superpone un gráfico de tipo stem con plt.stem(range(len(y)), y), resaltando los valores discretos de la señal.

---


## Correlación
La correlación mide la similitud entre dos señales en diferentes desplazamientos en el tiempo.
### Fórmula de la correlación cruzada:

$$
R_{xy}[n] = \sum_{k=-\infty}^{\infty} x[k] y[k+n]
$$

Donde:
- \(R_{xy}[n]\) es la correlación cruzada entre \(x\) y \(y\).
- \(x[k]\) y \(y[k+n]\) representan las señales en diferentes desplazamientos temporales.


### 1. Señal Cosenoidal
```python
Ts = 1.25e-3
n = np.arange(0, 9) #valores enteros
x1 = np.cos(2*np.pi*100*n*Ts)
fig = plt.figure(figsize=(10, 5)) 
plt.plot(n, x1, label="", color='black')
plt.title("Señal Cosenoidal")  
plt.xlabel("(n)") 
plt.ylabel("x1 [nTs]") 
plt.grid()
plt.stem(range(len(x1)), x1)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/ff850885-25c4-4218-9973-a7d8fdd429ea" alt="imagen" width="450">
</p>


Se genera y grafica una señal cosenoidal muestreada. Primero, se define un periodo de muestreo Ts = 1.25e-3, y luego se crea un arreglo n con valores enteros de 0 a 8 usando np.arange(0, 9). La función np.arange(inicio, fin) genera una secuencia de números desde inicio hasta fin-1 con un paso de 1 por defecto. En este caso, n representa los instantes de muestreo en el dominio discreto.

A partir de n, se calcula la señal x1 como un coseno de 100 Hz evaluado en los instantes n * Ts. Para la visualización, se crea una figura de tamaño 10x5, donde plt.plot(n, x1, color='black') traza la señal con una línea negra, y plt.stem(range(len(x1)), x1) resalta los valores discretos.

---

### 2. Señal Senoidal
```python
x2 = np.sin(2*np.pi*100*n*Ts)
fig = plt.figure(figsize=(10, 5)) 
plt.plot(n, x2, label="", color='black')
plt.title("Señal Senoidal")  
plt.xlabel("(n)") 
plt.ylabel("x2 [nTs]") 
plt.grid()
plt.stem(range(len(x2)), x2)
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/1ad296f4-c07c-4391-b529-f078c5ddc9b6" alt="imagen" width="450">
</p>

Al igual que en la gráfica anterior, este código genera y visualiza una señal, pero en este caso es una señal senoidal en lugar de una cosenoidal. Se usa el mismo conjunto de valores n = np.arange(0, 9), generado con np.arange(), y se calcula x2 como un seno de 100 Hz evaluado en los instantes n * Ts.

---

### 3. Correlación de las Señales y Representación Grafica
```python
correlacion = np.correlate(x1,x2,mode='full')
print('Correlación =',correlacion)
fig = plt.figure(figsize=(10, 5)) 
plt.plot(correlacion, color='black')
plt.stem(range(len(correlacion)), correlacion)
plt.title("Correlación")  
plt.xlabel("(n)") 
plt.ylabel("R[n]") 
plt.grid()
```
Se calcula y grafica la correlación cruzada entre las señales x1 y x2. La correlación mide la similitud entre dos señales a diferentes desplazamientos en el tiempo, lo que permite identificar patrones compartidos o desfases entre ellas.

Primero, np.correlate(x1, x2, mode='full') computa la correlación cruzada, generando una nueva señal correlacion, cuya longitud es len(x1) + len(x2) - 1. Luego, el resultado se imprime en la consola.

$$
\text{Correlación} = \begin{bmatrix}
-2.44929360 \times 10^{-16} & -7.07106781 \times 10^{-1} & -1.50000000 & -1.41421356 \\
-1.93438661 \times 10^{-16} & 2.12132034 \times 10^{0} & 3.50000000 & 2.82842712 \\
8.81375476 \times 10^{-17} & -2.82842712 \times 10^{0} & -3.50000000 & -2.12132034 \\
3.82856870 \times 10^{-16} & 1.41421356 \times 10^{0} & 1.50000000 & 7.07106781 \times 10^{-1} \\
0.00000000 \times 10^{0}
\end{bmatrix}
$$

<p align="center">
    <img src="https://github.com/user-attachments/assets/2616db03-294f-474f-81ec-f89dc7211d0e" alt="imagen" width="450">
</p>

Para visualizar la correlación, se crea una figura de 10x5 donde plt.plot(correlacion, color='black') dibuja la señal con una línea negra, mientras que plt.stem(range(len(correlacion)), correlacion) resalta sus valores discretos. 

---
## Transformación (Señal Electromiografica)
### 1. Caracterizacion en Función del Tiempo 
```python
datos = wfdb.rdrecord('session1_participant1_gesture10_trial1') 
t = 1500
señal = datos.p_signal[:t, 0] 
fs = datos.fs
```
Se carga una señal de electromiografía (EMG) y se extraen los primeros 1500 puntos.

#### 1.1. Estadisticos Descriptivos y frecuencia de muestreo
```python
def caracterizacion():
    print()
    print()
    media = np.mean(señal)
    desvesta = np.std(señal)
    print('Media de la señal:',np.round(media,6))
    print('Desviación estándar:',np.round(desvesta,6))
    print("Coeficiente de variación:",np.round((media/desvesta),6))
    print('Frecuencia de muestreo:',fs,'Hz')
    
    fig = plt.figure(figsize=(8, 4))
    sns.histplot(señal, kde=True, bins=30, color='black')
    plt.hist(señal, bins=30, edgecolor='blue')
    plt.title('Histograma de Datos')
    plt.xlabel('datos')
    plt.ylabel('Frecuencia')

caracterizacion()
```
- Media de la señal: 0.000131
- Desviación estándar: 0.071519
- Coeficiente de variación: 0.001834
- Histograma:
<p align="center">
    <img src="https://github.com/user-attachments/assets/f49fce8f-274a-47b9-bdb4-d45a9bab7513" alt="imagen" width="450">
</p>
- Frecuencia de muestreo: 2048 Hz

#### 1.2. Grafica de Electromiografía
```python
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal, color='m')
plt.title("Electromiografía [EMG]")  
plt.xlabel("muestras[n]") 
plt.ylabel("voltaje [mv]") 
plt.grid()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/a7661d06-f365-4edb-9084-1bd64b07475b" alt="imagen" width="450">
</p>



### 2. Descripción la señal en cuanto a su clasificación 

La señal electromiográfica (EMG) es un registro de la actividad eléctrica generada por los músculos esqueléticos. Se clasifica como una señal biomédica no estacionaria y altamente variable, influenciada por la activación muscular, la fatiga y factores externos como la calidad de los electrodos y el ruido ambiental. En términos generales, la EMG se puede analizar en el dominio del tiempo y la frecuencia para extraer características que ayuden en aplicaciones como el control de prótesis, diagnóstico de trastornos neuromusculares y análisis del rendimiento muscular en deportes.

### 3. Tranformada de Fourier
La transformada de Fourier permite convertir una señal del dominio del tiempo al dominio de la frecuencia.

### Fórmula de la Transformada de Fourier Discreta (DFT):

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi k n / N}
$$

Donde:

- \(X[k]\) es la representación en frecuencia de la señal.
- \(x[n]\) es la señal original en el dominio del tiempo.
- \(N\) es el número total de muestras.
- \(e^{-j 2 \pi k n / N}\) representa la base exponencial compleja.

La DFT utiliza una suma ponderada de las muestras de la señal con bases exponenciales complejas para transformar la señal desde el tiempo hacia el dominio de la frecuencia.

#### 3.1. Grafica de la transformada de fourier
El siguiente código muestra cómo calcular y graficar la transformada de Fourier de una señal:
```python
N = len(señal)
frecuencias = np.fft.fftfreq(N, 1/fs)
transformada = np.fft.fft(señal) / N
magnitud = (2 * np.abs(transformada[:N//2]))**2

plt.figure(figsize=(10, 5))
plt.plot(frecuencias[:N//2], np.abs(transformada[:N//2]), color='black')
plt.title("Transformada de Fourier de la Señal")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()
```
- np.fft.fft: Calcula la transformada de Fourier de la señal.
- np.fft.fftfreq: Devuelve las frecuencias correspondientes a cada componente de la transformada.
- N//2: Se utiliza para considerar únicamente las frecuencias positivas.
- plt.plot: Genera una gráfica de la magnitud de la transformada.

Esta gráfica muestra las frecuencias presentes en la señal y su magnitud asociada.

<p align="center">
    <img src="https://github.com/user-attachments/assets/1cc48cf6-16d7-4152-945e-5f280ec6a2b6" alt="imagen" width="450">
</p>


#### 3.2. Grafica de la densidad espectral
En la práctica, para señales discretas y de duración finita, la DEP se estima utilizando la transformada de Fourier discreta (DFT). Al calcular la DFT de una señal y normalizar adecuadamente, se obtiene una estimación de su densidad espectral de potencia. Esta estimación permite identificar las frecuencias predominantes y analizar cómo se distribuye la energía de la señal en el dominio de la frecuencia.
```python
plt.figure(figsize=(10, 5))
plt.plot(frecuencias[:N//2], magnitud, color='black')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.title('Densidad espectral de la señal')
plt.grid()

plt.show()
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/9a883eae-0c13-455a-9441-be09de4f1103" alt="imagen" width="450">
</p>
- La densidad espectral muestra cómo la potencia de la señal se distribuye entre las diferentes frecuencias.
- magnitud: Representa la potencia de cada frecuencia, calculada como el cuadrado de la magnitud de la transformada de Fourier.
Ambas gráficas son fundamentales para comprender el comportamiento de la señal en el dominio de la frecuencia. La primera da información sobre las frecuencias presentes, mientras que la segunda muestra cómo se distribuye la energía de la señal en esas frecuencias.

----
## Conclusión

Este laboratorio demuestra el uso de Python para el procesamiento digital de señales, aplicando herramientas matemáticas fundamentales como la convolución, la correlación y la transformada de Fourier para analizar una señal electromiográfica.
