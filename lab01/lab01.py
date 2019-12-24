from math import factorial as ft

import matplotlib.pyplot as plt
import numpy as np


def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)


eps = np.finfo(np.double).eps
print("Machine accuracy:", eps)


def f_div_mult(x, d=np.pi, n=52):
    for k in range(n): x = x / d
    for k in range(n): x = x * d
    return x


def f_sqrt_sqr(x, n=52):
    for k in range(n): x = np.sqrt(x)
    for k in range(n): x = x * x
    return x


def plot_error(x0, err):
    mask = np.logical_and(err > 0, err < np.inf)
    plt.loglog(x0[mask], err[mask], ".k")
    plt.loglog(x0, [eps] * len(err), "--r")  # машинная точность для сравнения
    plt.xlabel("$Значение\;аргумента$")
    plt.ylabel("$Относительная\;погрешность$")
    plt.show()


x0 = np.linspace(1.01, 1.95, 50, dtype=np.double)
'''x = f_sqrt_sqr(x0)
err = relative_error(x0, x)
plot_error(x0, err)
print("Errors", err[:4], "...")'''


def f_sqrt_sqr_interleave(x, n=52):
    for k in range(n):
        x = np.sqrt(x)
        x = x * x
    return x


'''x = f_sqrt_sqr_interleave(x0)
err = relative_error(x0, x)
plot_error(x0, err)
print("Errors", err[:4], "...")'''


class MyNumber(object):
    def __init__(self, zeta):
        """Конструктор принимает zeta, но обьект соответствует числу x=1+zeta."""
        self.zeta = zeta

    def __str__(self):
        """На экран выводится значение x, которое может быть менее точно,
        чем храниемое значение."""
        return "{}".format(self.to_float())

    def from_float(x):
        """Создает число со значением, равным x."""
        return MyNumber(x - 1)

    def to_float(self):
        """Преобразует число в формат с плавающей запятой"""
        return self.zeta + 1

    def __mul__(self, other):
        """Перезагрузка операции умножения."""
        # print(self.zeta + other.zeta + self.zeta * other.zeta)
        return MyNumber(self.zeta + other.zeta + self.zeta * other.zeta)
        # def __mul__(self, other):
        #    """Так делать нельзя."""
        #    return MyNumber.from_float(self.to_float()*other.to_float())

    def sqrt(self, N):
        s = 0.0
        for i in range(1, N):
            a = (-1) ** i * ft(2 * i)
            b = (1 - 2 * i) * ft(i) ** 2 * 4 ** i
            c = a / b
            #print(a, b, s)
            s += c * self.zeta ** i
        return MyNumber(s)


s = MyNumber(x0 - 1)


def f(x):
    x0 = x - 1
    return 1 + x0 / 2 - x0 ** 2 / 8 + x0 ** 3 / 16 - 5 * x0 ** 4 / 128


def g(s, n=52):
    print(s.zeta, x0)
    for k in range(n): s = s.sqrt(500)
    for k in range(n):
        # print(s.zeta)
        s = s * s
    return s.to_float()


x = g(s)
err = relative_error(x0, x)
plot_error(x0, err)

'''def f(x, n=52):
    for k in range(n):
        x = np.sqrt(x)
    return e ** (52 * np.log(x))


x = f(x0)
err = relative_error(x0, x)
plot_error(x0, err)
print("Errors", err[:4], "...")'''
