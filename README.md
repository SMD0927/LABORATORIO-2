# LABORATORIO-2
## Convolución, correlación y transformación 

En este laboratorio exploramos señales fisiológicas de ECG utilizando técnicas de estadística descriptiva y modelos de ruido. El objetivo es entender tanto las características propias de la señal como el impacto del ruido, analizando aspectos como la relación señal-ruido (SNR).

## Requisitos
- *Python 3.9*
- Bibliotecas necesarias:
  - wfdb
  - numpy
  - matplotlib
  - seaborn

Instalar dependencias:
bash
pip install wfdb numpy matplotlib seaborn


## Estructura del Código

### 1. Lectura de Datos
python
import wfdb
import numpy as np

datos = wfdb.rdrecord('rec_2')
t = 2000
señal = datos.p_signal[:t, 0]

Se utiliza wfdb.rdrecord para cargar una señal fisiológica (ECG) desde un archivo estándar en formato WFDB que fueron descargados en PhysioNet. En este caso, se seleccionan los primeros 2000 puntos de la señal. Este paso inicial permite trabajar con un subconjunto significativo de datos para realizar análisis detallados.

---

### 2. Histograma de la Señal
python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(señal, kde=True, bins=30, color='red')
plt.hist(señal, bins=30, edgecolor='blue')
plt.title('Histograma de Datos')
plt.xlabel('Datos')
plt.ylabel('Frecuencia')
plt.show()

<p align="center">
    <img src="https://i.postimg.cc/50qyPvY9/histograma.png" alt="histograma" width="450">
</p>

El histograma muestra una distribución asimétrica con mayor concentración de valores cerca de 0 y una cola extendida a la derecha, indicando un sesgo positivo. Esto sugiere la posible presencia de ruido o eventos atípicos en la señal, aunque la mayoría de los valores se mantienen dentro de un rango fisiológico típico.

---

### 3. Graficado de la Señal
python
plt.figure(figsize=(10, 5))
plt.plot(señal, label="Señal fisiológica")
plt.title("ECG")
plt.xlabel("TIEMPO [ms]")
plt.ylabel("VOLTAJE [mV]")
plt.legend()
plt.grid()
plt.show()

<p align="center">
    <img src="https://github.com/user-attachments/assets/d8104ccb-6b13-49c2-b510-abae7d5338f3" alt="image" width="500">
</p>


La gráfica muestra la señal ECG en función del tiempo, donde se pueden ver claramente las ondas características (P, QRS y T) y cómo varía el voltaje. Se aprecia un patrón cíclico que indica una actividad cardíaca regular, aunque también se observa algo de ruido en la línea base, lo que podría deberse a interferencias en la toma de datos.
---

### 4. Estadísticos Descriptivos

#### 4.1. Cálculo Manual
python
def estadisticos_programados():
    suma = 0
    for v in señal:
        suma += v    
    media = suma / t
    suma2 = sum((u - media)**2 for u in señal)
    desvesta = (suma2 / (t - 1))**0.5
    coeficiente = desvesta / media
    print('media:', media)
    print("desviacion estandar:", desvesta)
    print('coeficente de variacion', coeficiente)

estadisticos_programados()

Se calculan los siguientes estadísticos:
- *Media (μ):* Valor promedio de la señal.
- *Desviación Estándar (σ):* Medida de la dispersión de los datos respecto a la media.
- *Coeficiente de Variación (CV):* Relación entre desviación estándar y media, expresada en porcentaje.

$$
\mu = \frac{\sum x_i}{n}, \quad
\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n-1}}, \quad
CV = \frac{\sigma}{\mu}
$$


*Resultados:*
- Media: -0.0124
- Desviación estándar: 0.131
- Coeficiente de variación: -10.557

*Interpretación:*
La media cercana a cero indica una señal centrada, mientras que el coeficiente de variación muestra una variabilidad moderada.

#### 4.2. Usando Funciones de NumPy
python
def estadisticos_Bibliotecas():
    media = np.mean(señal)
    desvesta = np.std(señal, ddof=1)
    coeficiente = desvesta / media
    print('Media:', media)
    print("Desviación estándar:", desvesta)
    print('Coeficiente de variación:', coeficiente)

estadisticos_Bibliotecas()

Se obtienen los mismos resultados de manera más eficiente utilizando NumPy.

*Resultados:*
- Media: -0.012
- Desviación estándar: 0.131
- Coeficiente de variación: -10.554

---
En esta sección se calculan estadísticas básicas de la señal de dos formas: de manera manual y usando NumPy. Ambas aproximaciones generan resultados muy similares: una media cercana a cero (-0.0124 o -0.012) lo que coincide con el histograma que revela una concentración de valores alrededor de este punto ,y una desviación estándar de 0.131, lo que indica que la señal está centrada y presenta una dispersión moderada. El coeficiente de variación, cercano a 10.55, lo que refleja una variabilidad relativa en la señal.
### 5. Función de Probabilidad
python
def calcular_funcion_probabilidad(senal):
    valores_unicos = np.unique(señal)
    probabilidades = {}
    for valor in valores_unicos:
        probabilidades[valor] = np.sum(señal == valor) / len(señal)
    for valor, prob in probabilidades.items():
        print(f"Valor: {valor:.5f}, Probabilidad: {prob:.5f}")

calcular_funcion_probabilidad(señal)

$$
P(v) = \frac{\text{Frecuencia Absoluta de } v}{\text{Total de Valores}}
$$

Se calcula la probabilidad de ocurrencia de cada valor único en la señal. Esto ayuda a comprender cómo se distribuyen los valores específicos.

*Ejemplo de Resultados:*
- Valor: -0.28000, Probabilidad: 0.00050
- Valor: 0.00000, Probabilidad: 0.01650

*Análisis:*
La mayoría de los valores tienen baja probabilidad individual, lo que refleja la variabilidad natural de la señal.

---

### 6. Ruido Añadido y Cálculo de SNR
#### 6.1. Ruido Gaussiano
python
ruido = np.random.normal(0, 0.1, t)
señal_ruidosa = señal + ruido

El ruido gaussiano es un tipo de ruido aleatorio cuyas variaciones siguen una distribución normal. Se define por su media (0 en este caso) y su desviación estándar (0.1, que controla su intensidad). Es común en señales fisiológicas debido a la electrónica del sistema de adquisición y otras fuentes de interferencia aleatoria.

#### 6.2. Ruido de Impulso
python
prob_impulso = 0.08
impulsos = np.random.choice([0, 1], size=len(señal), p=[1-prob_impulso, prob_impulso])
amplitud_impulso = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido2 = impulsos * amplitud_impulso

Este ruido se caracteriza por picos abruptos y esporádicos en la señal, generados aquí con una probabilidad del 8% (prob_impulso = 0.08). La función np.random.choice determina en qué puntos aparecen los impulsos (1 o 0), y la amplitud se asigna aleatoriamente con valores de ±0.2. Este ruido suele deberse a interferencias externas o fallos en la transmisión de datos.

#### 6.3. Ruido Tipo Artefacto
python
prob_imp = 0.15
impul = np.random.choice([0, 1], size=len(señal), p=[1-prob_imp, prob_imp])
amplitud = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido3 = impul * amplitud

Este ruido representa alteraciones no deseadas en la señal causadas por errores en la adquisición, como movimientos del paciente o fallos en los electrodos. Es similar al ruido de impulso, pero con una mayor probabilidad de ocurrencia (prob_imp = 0.15). Se genera con la misma lógica de np.random.choice, agregando perturbaciones aleatorias.
