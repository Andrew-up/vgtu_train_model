import matplotlib.pyplot as plt
import numpy as np

# full_np = np.zeros((5, 5, 1)).astype(np.uint8)
# a = np.zeros((5, 5, 1)).astype(np.uint8)
#
# print(full_np.shape)
#
#
#
#
#
# plt.imshow(a)
# plt.show()

def gen_test():
    for i in [2,3,4,5,6]:
        yield i

print(next(gen_test()))
print(next(gen_test()))
print(next(gen_test()))
