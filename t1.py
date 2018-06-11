import numpy as np
from collections import deque

def getarr():
	return [1,1,1,1,1]

e1 = deque(maxlen=100)
s = getarr()
e1.append([1, 1, 2, 3])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([5, 1, 6, 7])
e1.append([7, 8, 8, 8])

print e1
n1 = np.asarray(e1)

print n1

s1 = n1[len(n1)-10:, (0, 2)]

print s1
#print s3
#
#a = np.vstack(s1)
#b = np.vstack(s3)
#
#print np.concatenate((a,b),axis=1)





