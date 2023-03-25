import  numpy as np

a = np.array([[22, 0], [2, 0],[1,2]])

a[a>1] = 255
print(type(a))

print(a)