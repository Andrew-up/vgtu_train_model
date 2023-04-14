import numpy as np
import torch
a = np.array([[0.5, 6, 0.1, 3, 0.4, 2, 3, 4],
              [0.5, 6, 0.1, 3, 0.4, 2, 3, 4]])

a[a < 0.5] = 0
print(a)
# [[0.5 6.  0.  3.  0.  2.  3.  4. ]
#  [0.5 6.  0.  3.  0.  2.  3.  4. ]]


h = torch.tensor(0.4444)

print(round(h.item(), 3))
# print(range(h.values(), 3))