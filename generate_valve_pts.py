from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from numpy.linalg import det
from scipy.stats import dirichlet
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import sys
sys.path.insert(0,"D:/Workspace/RV-Fitting/")
from slice import slice
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point 
import random 
plt.style.use("seaborn")

def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if poly.contains(random_point):
            points.append(random_point)

    return points

def dist_in_hull(points, n):
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = points[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
    sample = np.random.choice(len(vols), size = n, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = n))

ed_rv = np.loadtxt("trimmed_RVendo_20.txt")
# ed_rv = np.loadtxt("RVshape\\RVendo_0.txt")
# ed_rv = np.array([ed_rv[:,0],ed_rv[:,1],-ed_rv[:,2]]).T*1000

N = 4
slice(N, ed_rv)
X = np.array(slice.slices)
print(len(X))
top_layer = np.array(X[-1])

tv = []
pv = []
tv.append(top_layer[top_layer[:,0] >= max(top_layer[:,0]) - 37])
pv.append(top_layer[top_layer[:,0] <= max(top_layer[:,0]) - 47])

tv = np.array(tv)[0]
pv = np.array(pv)[0]

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(ed_rv[:,0],ed_rv[:,1],ed_rv[:,2])
# ax.scatter(X[-1][:,0],X[-1][:,1],X[-1][:,2])
ax.scatter(tv[:,0],tv[:,1],tv[:,2],s = 50,label = "Tricuspid Valve (TV)")
ax.scatter(pv[:,0],pv[:,1],pv[:,2],s = 50,label = "Pulmonary Valve (PV)")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
# plt.show()

tv_verts = []
for i in range(0,len(tv)):
    # tv_verts.append((tv[i,0],tv[i,1]))
    tv_verts.append([tv[i,0],tv[i,1],tv[i,2]])

poly_tv = Polygon(tv_verts)
# tv_points = random_points_within(poly_tv,200)
tv_points = dist_in_hull(np.array(tv_verts),800)



# tv_test = []
# for p in tv_points:
#     tv_test.append([p.x,p.y])

pv_verts = []
for i in range(0,len(pv)):
    # pv_verts.append((pv[i,0],pv[i,1]))
    pv_verts.append([pv[i,0],pv[i,1],pv[i,2]])


poly_pv = Polygon(pv_verts)
# pv_points = random_points_within(poly_pv,200)
pv_points = dist_in_hull(np.array(pv_verts),200)


# pv_test = []
# for p in pv_points:
#     pv_test.append([p.x,p.y])

# points_within_tv = np.array(tv_test)
# points_within_pv = np.array(pv_test)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(ed_rv[:,0],ed_rv[:,1],ed_rv[:,2])
# ax.scatter(tv[:,0],tv[:,1],max(tv[:,2])*np.ones((len(tv))),s = 50)
ax.scatter(tv_points[:,0],tv_points[:,1],tv_points[:,2],s = 50, label = 'Points Capping the TV')
ax.scatter(pv_points[:,0],pv_points[:,1],pv_points[:,2],s = 50,label = 'Points Capping the PV')
# ax.scatter(points_within_pv[:,0],points_within_pv[:,1],np.mean(pv[:,2])*np.ones((len(points_within_pv))),s = 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.ylabel('y')
# plt.zlabel('z')
plt.legend()
plt.show()

ed_rv = list(ed_rv)

for i in range(0,len(tv_points)):
    ed_rv.append([tv_points[i,0],tv_points[i,1],tv_points[i,2]])

for i in range(0,len(pv_points)):
    ed_rv.append([pv_points[i,0],pv_points[i,1],pv_points[i,2]])

print(len(ed_rv))

np.savetxt("processed_RVendo_20.txt",ed_rv)