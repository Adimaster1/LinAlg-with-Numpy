import numpy as np

# a= np.array([[2,-4,6],[0,1,1],[1,1,0]])
# print(a)
# b=np.array([[0],[-10],[5]])
# print(b)

# c=np.linalg.inv(a)
# print(c)

# result = np.matmul(c,b)
# print(result)

d= np.array([[0,2,2],
                  [2,0,2],
                  [2,2,0],])
print(d)

# e = np.zeros(4)
# print(e)

# result = np.linalg.solve(d,e) # does the inverse and matmul in single step
# print(result)


## finding eigen vectors for matrix
result = np.linalg.eig(d)
print(result)
