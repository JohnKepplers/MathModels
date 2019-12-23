from math import sin, cos

import numpy as np

base = 10


def exact_sin_sum(K):
    return .5 * (sin(K) - cos(.5) / sin(.5) * cos(K) + cos(.5) / sin(.5)) / K


def exact_sum(K):
    """Точное значение суммы всех элементов."""
    return 1.


def samples_sin(N):
    a = np.array([sin(i) / N for i in range(1, N + 1)])
    # создаем выборку объединяя части
    # перемешиваем элементы выборки и возвращаем
    return np.random.permutation(a)


def samples_abs(N):
    array = [sin(i) / N for i in range(1, N + 1)]
    array.sort(key=abs)
    a = np.array(array)
    # создаем выборку объединяя части
    # перемешиваем элементы выборки и возвращаем
    return a


def Kahan_sum(x):
    s = 0.0  # частичная сумма
    c = 0.0  # сумма погрешностей
    for i in x:
        y = i - c  # первоначально y равно следующему элементу последовательности
        t = s + y  # сумма s может быть велика, поэтому младшие биты y будут потеряны
        c = (t - s) - y  # (t-s) отбрасывает старшие биты, вычитание y восстанавливает младшие биты
        s = t  # новое значение старших битов суммы
    return s


def relative_error(x0, x):
    """Погрешность x при точном значении x0"""
    return np.abs(x0 - x) / np.abs(x)




def samples(K):
    """"Элементы выборки"."""
    # создаем K частей из base^k одинаковых значений
    parts = [np.full((base ** k,), float(base) ** (-k) / K) for k in range(0, K)]

    # создаем выборку объединяя части
    samples = np.concatenate(parts)
    # перемешиваем элементы выборки и возвращаем
    return np.random.permutation(samples)


def direct_sum(x):
    """Последовательная сумма всех элементов вектора x"""
    s = 0.
    for e in x:
        s += e
    return s


def number_of_samples(K):
    """Число элементов в выборке"""
    return np.sum([base ** k for k in range(0, K)])


K = 7  # число слагаемых
direct, kahan, sort = [], [], []
for i in range(10):
    x = samples(K)  # сохраняем выборку в массив



    exact_sum_for_x = exact_sum(K)  # значение суммы с близкой к машинной погрешностью


    direct_sum_for_x = direct_sum(x)
    direct.append(relative_error(exact_sum_for_x, direct_sum_for_x))
    sorted_x = x[np.argsort(x)]
    sorted_sum_for_x = direct_sum(sorted_x)
    sort.append(relative_error(exact_sum_for_x, sorted_sum_for_x))

    Kahan_sum_for_x = Kahan_sum(x)  # сумма всех элементов по порядку
    kahan.append(relative_error(exact_sum_for_x, Kahan_sum_for_x))


print('Алгоритм Кэхэна:', sum(kahan)/len(kahan))
print('Сортировка по возрастанию:', sum(sort)/len(sort))
print('Прямой проход:', sum(direct)/len(direct))
direct, kahan, sort, sorted_kahan_increase, abs_kahan = [], [], [], [], []
N = 1111111
for i in range(10):
    x = samples_sin(N)
    y = samples_abs(N)

    exact_sum_for_x = exact_sin_sum(N)  # значение суммы с близкой к машинной погрешностью
    Kahan_sum_for_x = Kahan_sum(x)
    direct_sum_for_x = direct_sum(x)
    direct.append(relative_error(exact_sum_for_x, direct_sum_for_x))
    kahan.append(relative_error(exact_sum_for_x, Kahan_sum_for_x))
    sorted_x = x[np.argsort(x)]
    sorted_sum_for_x = direct_sum(sorted_x)
    sorted_kahan = Kahan_sum(sorted_x)
    sort.append(relative_error(exact_sum_for_x, sorted_sum_for_x))
    sorted_kahan_increase.append(relative_error(exact_sum_for_x, sorted_kahan))
    abs_sort_kahan = Kahan_sum(y)
    abs_kahan.append(relative_error(exact_sum_for_x, abs_sort_kahan))

print('Для знакопеременной последовательности:')
print('Алгоритм Кэхэна:', sum(kahan)/len(kahan))
print('Сортировка по возрастанию:', sum(sort)/len(sort))
print('Прямой проход:', sum(direct)/len(direct))
print('Алгоритм Кэхэна с сортировкой по возрастанию:', sum(sorted_kahan_increase)/len(sorted_kahan_increase))
print('Алгоритм Кэхэна с сортировкой по возрастанию абсолютных значений:', sum(abs_kahan)/len(abs_kahan))

# параметры выборки
mean = 1e6  # среднее
delta = 1e-5  # величина отклонения от среднего


def samples2(N_over_two):
    """Генерирует выборку из 2*N_over_two значений с данным средним и среднеквадратическим
    отклонением."""
    x = np.full((2 * N_over_two,), mean, dtype=np.double)
    x[:N_over_two] += delta
    x[N_over_two:] -= delta
    # print(x)
    return np.random.permutation(x)


def exact_mean():
    """Значение среднего арифметического по выборке с близкой к машинной точностью."""
    return mean


def exact_variance():
    """Значение оценки дисперсии с близкой к машинной точностью."""
    return delta ** 2


x = samples2(1000000)
y = x[np.argsort(x)]
print("Размер выборки:", len(x))
print("Среднее значение:", exact_mean())
print("Оценка дисперсии:", exact_variance())
print("Ошибка среднего для встроенной функции:", relative_error(exact_mean(), np.mean(x)))
print("Ошибка дисперсии для встроенной функции:", relative_error(exact_variance(), np.var(x)))


def direct_mean(x):
    return direct_sum(x) / len(x)


def kahan_mean(x):
    return Kahan_sum(x) / len(x)


def kahan_second_var(x):
    # print('direct:',direct_mean(x ** 2) - direct_mean(x) ** 2)
    # print('kahan:', kahan_mean(x ** 2) - kahan_mean(x) ** 2)
    return kahan_mean(x ** 2) - kahan_mean(x) ** 2


def kahan_first_var(x):
    return kahan_mean((x - kahan_mean(x)) ** 2)


def direct_first_var(x):
    """Первая оценка дисперсии через последовательное суммирование."""
    return direct_mean((x - direct_mean(x)) ** 2)


def direct_second_var(x):
    """Вторая оценка дисперсии через последовательное суммирование."""
    # print('1:', direct_mean(x ** 2), '2:', direct_mean(x) ** 2)
    return direct_mean(x ** 2) - direct_mean(x) ** 2


def online_second_var(x):
    """Вторая оценка дисперсии через один проход по выборке"""
    m = x[0]  # накопленное среднее
    m2 = x[0] ** 2  # накопленное среднее квадратов
    for n in range(1, len(x)):
        # print(m, m2)
        m = (m * (n - 1) + x[n]) / n
        m2 = (m2 * (n - 1) + x[n] ** 2) / n
    return m2 - m ** 2


print('Ошибка для алгоритма Кэхэна', relative_error(exact_variance(), kahan_first_var(x)))


def welford(x):
    m, m2 = x[0], x[0] ** 2
    for n in range(1, len(x)):
        # print(m, m2, x[n])
        delta = x[n] - m
        m += m2 / n
        delta2 = x[n] - m
        m2 += delta * delta2
    return m2


def welford2(x):
    m, m2 = x[0], 0
    for n in range(1, len(x)):
        m2 += (n * x[n] - m) ** 2 / (n * (n + 1))
        m += x[n]
    return m / len(x), m2 / len(x)


print("Ошибка для алгоритма Уэлфорта:", relative_error(exact_variance(), welford2(x)[1]))
print("Ошибка второй оценки дисперсии для последовательного суммирования:",
      relative_error(exact_variance(), direct_second_var(x)))
print("Ошибка второй оценки дисперсии для однопроходного суммирования:",
      relative_error(exact_variance(), online_second_var(x)))
print("Ошибка первой оценки дисперсии для последовательного суммирования:",
      relative_error(exact_variance(), direct_first_var(x)))
