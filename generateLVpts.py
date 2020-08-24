import numpy as np
import random as rand
import math as m

# finds points within the vicinty of the maximum in the z component then
# stores the x and y points into a vector (with z-max)


def genPts(lv):

    zmax = np.max(lv[:, 2])
    location = np.where(lv == zmax)
    
    ridx = [
        rand.choice(location[0]),
        rand.choice(location[1]),
    ]
    
    foo1 = ridx[np.random.permutation(np.size(ridx))[0]]
    foo2 = ridx[np.random.permutation(np.size(ridx))[1]]
    foo3 = ridx[np.random.permutation(np.size(ridx))[0]]

    if len(ridx) <= 2:
        p1 = [
            (m.ceil(np.size(lv[:, 0]) * np.random.rand(1, 1))),
            (m.ceil(np.size(lv[:, 1]) * np.random.rand(1, 1))),
            zmax,
        ]
        p2 = [
            (m.ceil(np.size(lv[:, 0]) * np.random.rand(1, 1))),
            (m.ceil(np.size(lv[:, 1]) * np.random.rand(1, 1))),
            zmax,
        ]
        p3 = [
            (m.ceil(np.size(lv[:, 0]) * np.random.rand(1, 1))),
            (m.ceil(np.size(lv[:, 1]) * np.random.rand(1, 1))),
            zmax,
        ]
    else:
    	p1 = [foo1[0],foo1[1],zmax]
    	p2 = [foo2[0],foo2[1],zmax]
    	p3 = [foo3[0],foo3[1],zmax]


    return np.array([p1, p2, p3]) 

