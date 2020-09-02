import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import plotly.graph_objects as go 
from sympy import solve, symbols, pprint
import numpy as np 
plt.style.use('seaborn')
import json
import plotly

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
	print(type(plane_func1))

	plane_func2 = np.dot(pv_vec,P - pv_ctd)

	x = np.linspace(-100,100,11)
	y = np.linspace(-100,100,11)
	# ys = np.linspace(-100,100,11)

	zplane1 = solve(plane_func1,zs)[0]
	zplane2 = solve(plane_func2,zs)[0]

	z1 = []
	z2 = []
	for i in range(0,len(x)):
		z1.append(zplane1.subs(xs,x[i]).subs(ys,y[i]))
		z2.append(zplane2.subs(xs,x[i]).subs(ys,y[i]))
	
	# xs = []
	# ys = []

	# xs1 = []
	# ys1 = []
	plane_pts1 = np.array([list(x),list(y),z1]).T
	plane_pts2 = np.array([list(x),list(y),z2]).T
	print(plane_pts1)

	# plot normals 
	ts = symbols('ts')
	line1 = rw_ctd + ts*rw_vec
	print(line1[0])
	line2 = pv_ctd + ts*pv_vec

	t = np.linspace(-20,20,11)
	line1pts = []
	line2pts = []
	for i in range(0,len(t)):
		line1pts.append([line1[0].subs(ts,t[i]),line1[1].subs(ts,t[i]),line1[1].subs(ts,t[i])])
		line2pts.append([line2[0].subs(ts,t[i]),line2[1].subs(ts,t[i]),line2[1].subs(ts,t[i])])

	xx,yy = np.meshgrid(range(-100,100),range(-100,100))
	fig = plt.figure().gca(projection ='3d')
	fig.plot_surface(xx,yy,plane_pts1[:,2])
	plt.show()