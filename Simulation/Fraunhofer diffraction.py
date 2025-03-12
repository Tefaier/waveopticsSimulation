import numpy as np
import matplotlib.pyplot as plt

# Параметры
lambda_ = 500e-9  # Длина волны, м (500 нм)
a = 10e-6         # Ширина щели, м (10 мкм)
D = 100e-6        # Размер области моделирования, м
N = 1024          # Количество точек

# Создание сетки
dx = D / N
x = np.linspace(-D/2, D/2, N, endpoint=False)

# Апертура (щель)
aperture = np.zeros(N)
aperture[np.abs(x) <= a/2] = 1

# Вычисление Фурье-образа
field = np.fft.fftshift(np.fft.fft(aperture))
intensity = np.abs(field)**2

# Масштабирование оси для углового распределения
f = np.fft.fftshift(np.fft.fftfreq(N, dx))
sin_theta = lambda_ * f  # sinθ ≈ λ * пространственная частота

# Визуализация
plt.figure(figsize=(10, 4))
plt.plot(sin_theta, intensity / np.max(intensity))
plt.xlabel(r'$\sin\theta$')
plt.ylabel('Интенсивность (отн. ед.)')
plt.title('Дифракция Фраунгофера на одной щели')
plt.grid()
plt.xlim(-0.1, 0.1)  # Ограничение по оси X для наглядности
plt.show()

# plt.savefig('Fraunhofer diffraction.png')