import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree
plt.style.use("seaborn")

ed_rv = np.loadtxt("RVshape\\RVendo_20.txt")

edge1 = np.loadtxt("RVshape\\rv_edge1.txt",dtype = int) - 1 
edge2 = np.loadtxt("RVshape\\rv_edge2.txt",dtype = int) - 1 


fig = plt.figure()
ax = plt.axes(projection = '3d')
# ax.scatter(ed_rv[:,0],ed_rv[:,1],-ed_rv[:,2], color ='r')
edge1_pts = []
edge2_pts = []

for i in range(0,len(edge1)):
    edge1_pts.append([ed_rv[edge1[i],0],ed_rv[edge1[i],1],ed_rv[edge1[i],2]])
    # ax.scatter(ed_rv[edge1[i],0],ed_rv[edge1[i],1],ed_rv[edge1[i],2],s = 50,color = 'r')

for i in range(0,len(edge2)):
    edge2_pts.append([ed_rv[edge2[i],0],ed_rv[edge2[i],1],ed_rv[edge2[i],2]])
    # ax.scatter(ed_rv[edge2[i],0],ed_rv[edge2[i],1],ed_rv[edge2[i],2],s = 50,color = 'g')

# edge1_pts = np.array(edge1_pts)
# edge2_pts = np.array(edge2_pts)

# mean_x_tv = np.mean(edge1_pts[:,0])
# mean_y_tv = np.mean(edge1_pts[:,1])
# mean_z_tv = np.mean(edge1_pts[:,2])

# mean_x_pv = np.mean(edge2_pts[:,0])
# mean_y_pv = np.mean(edge2_pts[:,1])
# mean_z_pv = np.mean(edge2_pts[:,2])

# ed_rv = list(ed_rv)

# ed_rv.append([mean_x_pv,mean_y_pv,mean_z_pv])
# ed_rv.append([mean_x_tv,mean_y_tv,mean_z_tv])

# np.savetxt("test_RVendo_0.txt",ed_rv)

# find a distance where the PV and TV have a point lying on a flat plane then cut off up to that point

tree1 = KDTree(ed_rv,leafsize= ed_rv.shape[0]+1)
k = 50
neighbor_dists, neighbor_idx1 = tree1.query(edge1_pts, k)
tv_nodes = []
for j in range(0,k):
    for i in range(0,len(neighbor_idx1)):
        # ax.scatter(ed_rv[neighbor_idx1[i,j],0],ed_rv[neighbor_idx1[i,j],1],ed_rv[neighbor_idx1[i,j],2],s = 50,color = 'r')
        tv_nodes.append(neighbor_idx1[i,j])

np.savetxt("tv_nodes.txt",tv_nodes)

tree2 = KDTree(ed_rv,leafsize= ed_rv.shape[0]+1)

neighbor_dists, neighbor_idx2 = tree2.query(edge2_pts, k)

pv_nodes = []
for j in range(0,k):
    for i in range(0,len(neighbor_idx2)):
        # ax.scatter(ed_rv[neighbor_idx2[i,j],0],ed_rv[neighbor_idx2[i,j],1],ed_rv[neighbor_idx2[i,j],2],s = 50,color = 'g')
        pv_nodes.append(neighbor_idx2[i,j])

np.savetxt("pv_nodes.txt",pv_nodes)

nodes_to_trim = list(np.concatenate((tv_nodes,pv_nodes)))
nodes_to_trim = list(set(nodes_to_trim))
temp = np.delete(ed_rv,nodes_to_trim,axis = 0)
print(type(temp))

x = temp[:,0]
y = temp[:,1]
z = temp[:,2]

trimmed_ed_rv = np.array([x,y,-z]).T*1000
print(trimmed_ed_rv)


ax.scatter(trimmed_ed_rv[:,0],trimmed_ed_rv[:,1],trimmed_ed_rv[:,2],color = 'b')
plt.axis("off")
plt.show()

np.savetxt("trimmed_RVendo_20.txt",trimmed_ed_rv)



