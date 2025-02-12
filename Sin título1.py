import numpy as np
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns

def samuel():
    print('datos de samuel')
    h = [5,6,0,0,8,3,9]
    x = [1,1,0,8,3,3,3,6,9,3]
    y = np.convolve(x,h,mode='full')
    print('h[n] =', h)
    print('x[n] =',x)
    print('y[n] =',y)

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(h,color='black')
    plt.stem(range(len(h)), h)
    plt.title("Sistema (samuel)")  
    plt.xlabel("(n)") 
    plt.ylabel("h [n]") 
    plt.grid() 

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(x,color='black')
    plt.stem(range(len(x)), x)
    plt.title("Señal (samuel)")  
    plt.xlabel("(n)") 
    plt.ylabel("x [n]") 
    plt.grid() 

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(y,color='black')
    plt.title("Señal Resultante (samuel)")  
    plt.xlabel("(n)") 
    plt.ylabel("y [n]") 
    plt.grid() 
    plt.stem(range(len(y)), y)
    print()
samuel()
def ana():
    print('datos de ana')
    h = [5,6,0,0,7,7,2]
    x = [1,0,7,0,0,0,6,8,7,2]
    y = np.convolve(x,h,mode='full')
    print('h[n] =', h)
    print('x[n] =',x)
    print('y[n] =',y)

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(h, color='b')
    plt.stem(range(len(h)), h)
    plt.title("Sistema (ana)")  
    plt.xlabel("(n)") 
    plt.ylabel("h [n]") 
    plt.grid() 

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(x,color='b')
    plt.stem(range(len(x)), x)
    plt.title("Señal (ana)")  
    plt.xlabel("(n)") 
    plt.ylabel("x [n]") 
    plt.grid() 

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(y,color='b')
    plt.title("Señal Resultante (ana)")  
    plt.xlabel("(n)") 
    plt.ylabel("y [n]") 
    plt.grid() 
    plt.stem(range(len(y)), y)
    print()
ana()
def santiago():
    print("datos de santigo")
    h = [5,6,0,0,7,7,5]
    x = [1,0,1,4,6,6,0,7,0,8]
    y = np.convolve(x,h,mode='full')
    print('h[n] =', h)
    print('x[n] =',x)
    print('y[n] =',y)

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(h,color='g')
    plt.stem(range(len(h)), h)
    plt.title("Sistema (santiago)")  
    plt.xlabel("(n)") 
    plt.ylabel("h [n]") 
    plt.grid() 

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(x,color='g')
    plt.stem(range(len(x)), x)
    plt.title("Señal (santiago)")  
    plt.xlabel("(n)") 
    plt.ylabel("x [n]") 
    plt.grid() 

    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(y,color='g')
    plt.title("Señal Resultante (santiago)")  
    plt.xlabel("(n)") 
    plt.ylabel("y [n]") 
    plt.grid() 
    plt.stem(range(len(y)), y)
    print()
    print()
santiago()


def b():
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

    x2 = np.sin(2*np.pi*100*n*Ts)
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(n, x2, label="", color='black')
    plt.title("Señal Senoidal")  
    plt.xlabel("(n)") 
    plt.ylabel("x2 [nTs]") 
    plt.grid()
    plt.stem(range(len(x2)), x2)

    print()
    correlacion = np.correlate(x1,x2,mode='full')
    print('Correlación =',correlacion)
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(correlacion, color='black')
    plt.stem(range(len(correlacion)), correlacion)
    plt.title("Correlación")  
    plt.xlabel("(n)") 
    plt.ylabel("R[n]") 
    plt.grid()
b()



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

datos = wfdb.rdrecord('session1_participant1_gesture10_trial1') 
t = 1500
señal = datos.p_signal[:t, 0] 
fs = datos.fs
caracterizacion()
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal, color='m')
plt.title("Electromiografia [EMG]")  
plt.xlabel("muestras[n]") 
plt.ylabel("voltaje [mv]") 
plt.grid()

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

plt.figure(figsize=(10, 5))
plt.plot(frecuencias[:N//2], magnitud, color='black')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.title('Densidad espectral de la señal')
plt.grid()

plt.show() 