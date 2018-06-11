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

print list(e1)
#n1 = np.asarray(e1)
#
#print n1
#
#s1 = n1[len(n1)-10:len(n1), 0]
#s3 = n1[len(n1)-10:len(n1), 2]
#
#print s1
#print s3
#
#a = np.vstack(s1)
#b = np.vstack(s3)
#
#print np.concatenate((a,b),axis=1)

#h3 = np.zeros(shape=(10, 3), dtype=np.float)
#h3 = np.zeros(10)
#print h3
#
#h3 = np.array([1,2,3,4])
#print h3


print 1.*1/float(100)-1.



