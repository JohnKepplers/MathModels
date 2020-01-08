from math import pi

import matplotlib.pyplot as plt
import numpy as np


def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)


def log_teylor_series(x, N=5):
    print(N)
    a = x - 1
    a_k = a  # x в степени k. Сначала k=1
    y = a  # Значене логарифма, пока для k=1.
    for k in range(2, N):  # сумма по степеням
        a_k = -a_k * a  # последовательно увеличиваем степень и учитываем множитель со знаком
        y = y + a_k / k
    return y


def get_N(eps):
    return (1 - eps) / eps


# Узлы итерполяции
N = 5
xn = 1 + 1. / (1 + np.arange(N))
zn = np.arange(N)
print(zn)
un = 3 * (xn - 1) / (2 * (1 + xn))


def opt_u(a, b, zn):
    return (a + b) / 2 + (b - a) / 2 * np.cos(((2 * zn + 1) * pi) / (2 * (zn + 1)))

opt = opt_u(0, 1, zn)
print(opt)
print(xn)
print(un)
yn = np.log(xn)
yyn = np.log(un)
optyn = np.log(opt)
# Тестовые точки
x = np.linspace(xn[4], xn[0], 1000)
xu = np.linspace(un[4], un[0], 1000)
xuu = np.linspace(opt[4], opt[0], 1000)
y = np.log(x)
yy = np.log(xu)
yyy = np.log(xuu)
# Многочлен лагранжа
import scipy.interpolate

L = scipy.interpolate.lagrange(xn, yn)
L1 = scipy.interpolate.lagrange(un, yyn)
Lopt = scipy.interpolate.lagrange(opt, optyn)
yl = L(x)
y2 = L1(xu)
y3 = Lopt(xuu)
plt.plot(x, y, '-k')
plt.plot(xn, yn, '.b')
plt.plot(x, yl, '-r')
#plt.plot(xu, y2, '-g')
#plt.plot(x, y3)
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()

plt.plot(xu, yy, '-k')
plt.plot(un, yyn, '.b')
plt.plot(xu, y2, '-g')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()



plt.plot(xuu, yyy, '-k')
plt.plot(opt, optyn, '.b')
plt.plot(xuu, y3, '-g')
plt.xlabel("$x$")
plt.ylabel("$y=\ln x$")
plt.show()

plt.semilogy(x, relative_error(y, yl), '-r')
plt.semilogy(xu, relative_error(yy, y2), '-g')
plt.semilogy(x, relative_error(yyy, y3), '-b')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.show()
