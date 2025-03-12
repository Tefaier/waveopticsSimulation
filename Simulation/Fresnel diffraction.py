import numpy as np
import matplotlib.pyplot as plt

# Параметры
lambda_ = 500e-9  # Длина волны, м (500 нм)
a = 10e-6         # Ширина щели, м (10 мкм)
z = 50e-3         # Расстояние до экрана, м (50 мм)
N = 2048          # Количество точек
dx = 0.5e-6       # Шаг сетки (0.5 мкм для лучшего разрешения)

# Апертура
x = np.linspace(-N*dx/2, N*dx/2, N, endpoint=False)
aperture = np.zeros(N)
aperture[np.abs(x) <= a/2] = 1.0

# Угловой спектр (FFT)
k = 2 * np.pi / lambda_
f_x = np.fft.fftshift(np.fft.fftfreq(N, dx))
FX = np.fft.fftshift(f_x)
spectrum = np.fft.fftshift(np.fft.fft(aperture)) * dx

# Точная передаточная функция
kx = 2 * np.pi * FX
kz = np.sqrt(k**2 - kx**2, where=(k**2 >= kx**2), out=np.zeros_like(kx))
H = np.exp(1j * kz * z)

# # Защита от отрицательных значений под корнем
# kz = np.sqrt(k**2 - (2*np.pi*FX)**2)  # k_z = sqrt(k² - k_x²)
# kz = np.where(np.isreal(kz), kz, 0)    # Обнуляем нефизичные (мнимые) компоненты
# H = np.exp(1j * kz * z)

# Поле на экране
field = np.fft.ifft(np.fft.ifftshift(spectrum * H)) * (N*dx)
intensity = np.abs(field)**2

# Визуализация
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("Апертура")
plt.plot(x, aperture)
plt.xlabel("x, м")

plt.subplot(122)
plt.title(f"Дифракция Френеля (z={z:.0e} м)")
plt.plot(x, intensity / np.max(intensity))
plt.xlabel("x, м")
plt.ylabel("Интенсивность (отн. ед.)")
plt.grid()
plt.tight_layout()
plt.show()

# plt.savefig('Fresnel diffraction.png')