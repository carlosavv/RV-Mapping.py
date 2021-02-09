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
	return ax

def plot_rv(rv):
	# fig = plt.figure(dpi = 150)
	# ax = plt.axes(projection = '3d')
	# ax.scatter(rv[:,0],rv[:,1],rv[:,2])
	# plt.show()
	# return ax

	fig = go.Figure(data = [go.Scatter3d(x = rv[:,0],y = rv[:,1],z = rv[:,2],mode = 'markers')])
	fig.show()

# def computeZ(func):

	# return zs = 

def plot_endPlanes(rv,pv_vec,rw_vec,pv_ctd,rw_ctd):
	# # plot planes

	# xs,ys,zs = symbols('xs ys zs')
	# P = [xs,ys,zs]
	# plane_func1 = np.dot(rw_vec,P - rw_ctd)
	# plane_func2 = np.dot(pv_vec,P - pv_ctd)

	# x = np.linspace(-100,100,101)
	# y = np.linspace(-100,100,101)
	# # ys = np.linspace(-100,100,11)

	# zplane1 = solve(plane_func1,zs)[0]
	# zplane2 = solve(plane_func2,zs)[0]

	# z1 = []
	# z2 = []
	# for i in range(0,len(x)):
	# 	z1.append(zplane1.subs(xs,x[i]).subs(ys,y[i]))
	# 	z2.append(zplane2.subs(xs,x[i]).subs(ys,y[i]))
	
	# # xs = []
	# # ys = []

	# # xs1 = []
	# # ys1 = []
	# plane_pts1 = np.array([list(x),list(y),z1]).T
	# plane_pts2 = np.array([list(x),list(y),z2]).T
	# print("plane_pts: ")
	# print(plane_pts1)

	# plot normals 
	ts = symbols('ts')
	line1 = rw_ctd + ts*rw_vec.T
	line2 = pv_ctd + ts*pv_vec.T

	t = np.linspace(-20,20,3)
	line1pts = []
	line2pts = []
	for i in range(0,len(t)):
		line1pts.append([line1[0].subs(ts,t[i]),line1[1].subs(ts,t[i]),line1[2].subs(ts,t[i])])
		line2pts.append([line2[0].subs(ts,t[i]),line2[1].subs(ts,t[i]),line2[2].subs(ts,t[i])])

	normal_line1 = np.array([line1pts[0],line1pts[1],line1pts[2]])
	normal_line2 = np.array([line2pts[0],line2pts[1],line2pts[2]])

	p1 = rw_ctd
	normal1 = rw_vec

	p2 = pv_ctd
	normal2 = pv_vec

	d1 = -np.sum(p1*normal1)
	d2 = -np.sum(p2*normal2)

	# create x,y
	print(type(round(max(rv[:,0]))))
	xx, yy = np.meshgrid(range(int(round(min(rv[:,0]))),int(round(max(rv[:,0])))),range(int(round(min(rv[:,1]))),int(round(max(rv[:,1])))))

	# calculate corresponding z
	z1 = (-normal1[0]*xx - normal1[1]*yy - d1)*1./normal1[2]
	z2 = (-normal2[0]*xx - normal2[1]*yy - d2)*1./normal2[2]

	# plot the surface
	plt3d = plt.figure().gca(projection='3d')
	plt3d.scatter(rv[:,0],rv[:,1],rv[:,2])
	plt3d.plot(normal_line1[:,0],normal_line1[:,1],normal_line1[:,2])
	plt3d.plot(normal_line2[:,0],normal_line2[:,1],normal_line2[:,2])
	plt3d.plot_surface(xx,yy,z1, color='red')
	plt3d.plot_surface(xx,yy,z2, color='green')
	return plt3d

def plot_bezier(rv,p0,p1,p2,p3):
	t = np.linspace(0,1,200)

	B = np.zeros((len(t),3))

	for i in range(0,len(t)):
		B[i,:] = (1-t[i]**3)*p0 + 3*(1-t[i]**2)*t[i]*p1 + 3*(1-t[i])*t[i]**2*p2 + t[i]**3*p3
	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter(rv[:,0],rv[:,1],rv[:,2])
	ax.plot(B[:,0],B[:,1],B[:,2])
	return ax