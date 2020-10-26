import numpy as np
a = []
a.append([1,0.,1])
a.append([1,1.,1])
b= np.reshape([0,1],[2,1])
a = np.reshape(a,[2,3])
print(a)


print(a.shape)
print(b.shape)
print(a[b[:,0]!=0])

