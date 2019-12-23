import matplotlib.pyplot as plt
import numpy as np

eps = 1e-3
n = 100/eps
x = np.linspace(1, 1 + eps, 100)
print(x)
y = np.log(x)
plt.semilogx(x, y)
plt.semilogx(x ** (-n), -n * y)
plt.semilogx(x ** n, n * y)
plt.xlabel('$x$')
plt.ylabel('$y=\ln x$')
plt.show()
