import numpy as np 
import scipy as sp 


def affine_fit(X):

    # this implementation isn't the most efficient
    # TODO:
    '''
    fix issue with using eig() 

    issue: inconsistencies with return parameter v - 3x3 matrix of eigenvectors  
    '''
    
    p = np.mean(X,axis = 0)
    # pmeany = np.mean(X,axis =1)
    # pmeanz = np.mean(X,axis=2)
    # p = np.array([[pmeanx,pmeany,pmeanz]])
    R = X - p
    # print("R = ", R)
    t = np.matmul(R.T, R)
    w,v = np.linalg.eig(t)

    # might be best to just return whole matrix v and see how that works instead

    # print("w = \n", w)
    # print('')
    # print("v = \n", v)
    # print('')
    n = np.array(v[:,2])
    # print("n = \n", n)
    # print('')
    V = np.array(v[:,0])
    # print("V = \n", V)
    # print('')
    # t = np.array(v[2][:])

    return [n,V,p]