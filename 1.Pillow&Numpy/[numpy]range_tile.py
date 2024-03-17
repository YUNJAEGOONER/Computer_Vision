import numpy as np


#test arange & tile
# array = [0 2 4 6 8]
array = np.arange(0, 10, 2)

# [0 2 4 6 8]
# [0 2 4 6 8]
# [0 2 4 6 8]

array_test = np.tile(array, [3, 1])
print(array_test)

print("----------------------------")

# [0 2 4 6 8 0 2 4 6 8]
# [0 2 4 6 8 0 2 4 6 8]
# [0 2 4 6 8 0 2 4 6 8]

array_test_2 = np.tile(array, [3,2])
print(array_test_2)