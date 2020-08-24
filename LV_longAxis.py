import numpy as np
from generateLVpts import genPts
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from scipy.spatial import ConvexHull
import scipy as sp
import sys
np.set_printoptions(threshold=sys.maxsize)
from matplotlib.path import Path 
from skimage.measure import label, regionprops
from skimage.draw import polygon2mask 

'''
functions that find the center of the mitral valve (mv)
determines the location of the apex

using the mvcenter and apex the LV's long axis is computed
'''

def compute_lvLA(P, lv):
    p1 = P[0]
    p2 = P[1]
    p3 = P[2]
    mvcenter = p2
    cprod = np.cross((p1 - p2), (p3-p2))
    n = cprod/np.linalg.norm(cprod)

    for i in range(0, 1):
        thickness = 2
        count = 0
        top_plane = []
        
        for j in range(0, len(lv)):
            if abs(np.dot(n, mvcenter-lv[j, :])) < thickness:
                top_plane.append(lv[j, :])
        # evaluate LV's top plane and fit a plane on to the data points
        top_plane = np.array(top_plane)
        [n, V, mvcenter] = affine_fit(top_plane)
        n = np.array(n.T)
        tngnt = n
        node_pt = mvcenter

        if tngnt[2] != 0:
            pt1 = [1, 1, (-tngnt[0]-tngnt[1] +
                          np.dot(tngnt, node_pt))/tngnt[2]]
        elif tngnt[1] != 0:
            pt1 = [1, (-tngnt[0]-tngnt[2] +
                       np.dot(tngnt, node_pt))/tngnt[1], 1]
        else:
            pt1 = [(-tngnt[1]-tngnt[2]+np.dot(tngnt, node_pt)) /
                    tngnt[0], 1, 1]

        # Find two inplane vectors to take projections
        tangent = np.array(tngnt/(np.sqrt(np.dot(tngnt,tngnt))))
        v1 = np.array(pt1 - node_pt)
        v2 = np.cross(tangent, v1)
        v1n = v1/np.linalg.norm(v1)
        v2n = v2/np.linalg.norm(v2)

        count = 0
        proj = []

        # iterate through the LV data and store projections
        for j in range(len(lv)):
            sample = lv[j, :]
            proj.append(proj_on_plane(
                tangent, X= np.array([v1n, v2n]), a=np.array([node_pt, sample])))
            count += 1

        proj = np.array(proj)
        hull = ConvexHull(proj)
        xtrans = min(proj[hull.vertices,0])
        ytrans = min(proj[hull.vertices,1])

        fig = plt.figure()

        plt.plot(proj[hull.vertices,0],proj[hull.vertices,1])

        projections = []
        for ii in range(0,len(hull.vertices)):
            projections.append([proj[hull.vertices[ii],0] - xtrans+5,
                                proj[hull.vertices[ii],1] - ytrans+5])

        projections = np.array(projections)
        print(projections)

        image_shape = (250,250)
        polygon = projections
        mask = polygon2mask(image_shape,polygon)
        print(mask)
        img = mask
        label_img = label(img)
        regions = regionprops(label_img)
        for props in regions:
            xcentroid = props.centroid[0]
            ycentroid = props.centroid[1]
        # xcentroid = np.mean(proj[:,0]) + xtrans-5
        # ycentroid = np.mean(proj[:,1]) + ytrans-5


        proj_ctd = np.array([xcentroid+xtrans-5, ycentroid+ytrans-5])
        mvcenter = proj_ctd[0]*v1n + proj_ctd[1]*v2n + node_pt

    apex = generateApexPts(lv)

    long_axisv = mvcenter - apex

    long_axis = long_axisv/np.linalg.norm(long_axisv)
    print(long_axis)
    # print('apex1',apex)
    return long_axis,mvcenter,apex

def proj_on_plane(n, X, a):

    alpha = np.dot((a[0]-a[1]), n)/np.dot(n, n)
    q = a[1] + alpha*n
    b = ((X[0, 0]*(q[1]-a[0, 1])) - (X[0, 1]*(q[0]-a[0, 0]))) / \
         (X[0, 0]*X[1, 1] - X[0, 1]*X[1, 0])
    avec = (q[0] - a[0, 0] - b*X[1, 0])/X[0, 0]

    return np.array([avec, b])


def affine_fit(X):

    p = np.mean(X, axis=0)
    # pmeany = np.mean(X,axis =1)
    # pmeanz = np.mean(X,axis=2)
    # p = np.array([[pmeanx,pmeany,pmeanz]])
    R = np.array(X[...,:] - p)
    t = np.dot(R.T, R)
    [w, v] = np.linalg.eig(t)
    n = np.array(v[:,2])
    V = np.array(v[:, 1:])
    return [n, V, p]


def generateApexPts(lv):
    zmin = np.min(lv[:, 2])
    location = np.where( lv == zmin)
    # print('locations',location)
    # print(len(location))
    mx = np.zeros((len(location),2))
    for i in location:
        mx = [lv[i,0],lv[i,1]]
    xctd = np.mean(mx[0])
    yctd = np.mean(mx[1])

    apex = [xctd,yctd,zmin]
    return apex



