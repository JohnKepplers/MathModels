import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)


def log_newton(x, eps = 6.66133814776e-16):
    y = 1 # начальное приближение
    while True:
        diff = -1 +  x / np.exp(y)
        '''print(diff)
        print('______________________________')'''
        a, b = min(diff), max(diff)
        m = max(abs(a), abs(b))
       #print(m, np.sqrt(2*np.exp(-10)*eps/np.exp(10)))
        if m < eps:
            break
        y = y - 1 + x / np.exp(y)
    return y


x = np.logspace(-3, 3, 1000)
y0 = np.log(x)
y = log_newton(x)
plt.loglog(x, relative_error(y0, y), '-k')
plt.xlabel("$Аргумент$")
plt.ylabel("$Относительная\;погрешность$")
plt.show()
