import numpy as np

a = np.linspace(0, 1, 3, endpoint=False)
b = np.linspace(1, 0, 4, endpoint=True)
print(a)
print(b)

a1 = np.linspace(0, 1, 4, endpoint=False)
b1 = np.linspace(1, 0, 5, endpoint=True)

c = np.concatenate([a, b,])
print(c)
c1 = np.concatenate([a1, b1])

d = c[1:-1]
print(d)
d1 = c1[1:-1]
# e = c[1:-1]
# e1 = c1[1:-1]
print(d1)

f = np.outer(d, d1)
print(f)