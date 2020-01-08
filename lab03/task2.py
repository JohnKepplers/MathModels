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


x0 = np.logspace(-5, 5, 1000, dtype=np.double)
epsilon = np.finfo(np.double).eps
best_precision = (epsilon / 2) * np.abs(1. / np.log(x0))

x = np.logspace(-5, 1, 1001)
y0 = np.log(x)
y = log_teylor_series(x, N = int(get_N(10e-3)))
plt.loglog(x, relative_error(y0, y), '-k')
plt.loglog(x0, best_precision, '--r')
plt.xlabel('$x$')
plt.ylabel('$(y-y_0)/y_0$')
plt.legend(["$Достигнутая\;погр.$", "$Минимальная\;погр.$"], loc=5)
plt.show()
