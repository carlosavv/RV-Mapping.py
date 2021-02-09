import numpy as np 

'''
args: 

vrw: normal of right wall 
vpv: normal of pulmonary valve 

ctd_rw: centroid of right wall
ctd_pv: centroid of pulmonary valve

g1: guess parameter 1 
g2: guess parameter 2 

returns:

first approximation of bezier curve (control points)
'''

def guess_CA(vrw, vpv, ctd_rw, ctd_pv, g1, g2):
    
    p0 = ctd_rw
    p3 = ctd_pv
    normal1 = vrw 
    normal4 = vpv 

    A_12 = np.array([
        [normal1[1],-normal1[0],0],
        [normal1[2],0,-normal1[0]],
        [1,0,0]]
    )

    bezier_12 = np.array(
        [   normal1[1]*p0[0] - normal1[0]*p0[1],
            -normal1[0]*p0[2] + normal1[2]*p0[0],
            g1
        ]
    )

    A_34 = np.array(
        [[normal4[1],-normal4[0],0],
        [normal4[2],0,-normal4[0]],
        [1,0,0]]
    )
    
    bezier_34 = np.array(
        [   normal4[1]*p3[0] - normal4[0]*p3[1],
            -normal4[0]*p3[2] + normal4[2]*p3[0],
            g2
        ]
    )

    p1 = np.matmul(np.linalg.inv(A_12),bezier_12)
    p2 = np.matmul(np.linalg.inv(A_34),bezier_34)
    
    return p0,p1,p2,p3


