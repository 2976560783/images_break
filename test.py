import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a)
print(a.shape)
a.resize((6, ))
print(a)
print(a.shape)
a.resize((2, 3))
print(a)