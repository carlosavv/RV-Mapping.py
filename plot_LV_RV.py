import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import mplcursors
import plotly.graph_objects as go 
from sympy import solve, symbols, pprint
import numpy as np 
plt.style.use('seaborn')

def plot_lv_rv(lv,rv,mvcenter,apex):
    
    fig = plt.figure(dpi = 150)
    ax = plt.axes(projection = '3d')
    ax.scatter(lv[:,0],lv[:,1],lv[:,2])
    ax.scatter(rv[:,0],rv[:,1],rv[:,2])
    ax.plot([mvcenter[0],apex[0]],[mvcenter[1],apex[1]],[mvcenter[2],apex[2]],color = 'r', linewidth = 5)

def plot_rv(rv):
	# fig = plt.figure(dpi = 150)
	# ax = plt.axes(projection = '3d')
	# ax.scatter(rv[:,0],rv[:,1],rv[:,2],picker= True)
	# return ax

	fig = go.Figure(data = [go.Scatter3d(x = rv[:,0],y = rv[:,1],z = rv[:,2],mode = 'markers')])
	fig.show()

# def computeZ(func):

	# return zs = 

def plot_endPlanes(pv_vec,rw_vec,pv_ctd,rw_ctd):
	# plot planes

	xs,ys,zs = symbols('xs ys zs')
	P = [xs,ys,zs]
	plane_func1 = np.dot(rw_vec,P - rw_ctd)
	print(plane_func1)
	plane_func2 = np.dot(pv_vec,P - pv_ctd)

	zplane1 = solve(plane_func1,zs)
	print(type(zplane1))
	zplane2 = solve(plane_func2,zs)

	# plot normals 
	ts = symbols('ts')
	line1 = rw_ctd + ts*rw_vec
	line2 = pv_ctd + ts*pv_vec

