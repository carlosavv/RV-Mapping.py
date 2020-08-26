import numpy as np 
from LV_longAxis import affine_fit
from sympy import solve, symbols, pprint
from plot_LV_RV import plot_rv
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0,'C:/Workspace/RV-Fitting/')
from slice import slice
# generates points that fit a plane at the PV

def genPVpts(rv):

	# plot_rv(rv)
	# candidate_pt = np.array(input("Enter a point [x,y,z] located at the PV: "))
	# print(type(candidate_pt))
	
	print('Enter x,y,z coordinates located at the PV: ')
	
	n = 3
	candidate_pt = []
	for i in range(0,n):
		candidate_pt.append(float(input()))

	# N = 5

	# slice(N,rv)

	# lowerBound = slice.slices[0]
	# upperBound = slice.slices[-1]

	# lowerBound =  

	lowerBound = np.array(rv[rv[:,2] == min(rv[:,2])][0])
	print(type(lowerBound))
	vpv = np.subtract(candidate_pt,lowerBound)
	vpv = vpv/np.linalg.norm(vpv)



	ctd_pv = candidate_pt

	tol = 5 
	pv = []
	for i in range(0,len(rv)):
		res = np.dot(vpv,rv[i]-ctd_pv)

		if abs(res) <= tol:
			pv.append([rv[i,0],rv[i,1],rv[i,2]])

	[vpv,foo,ctd_pv] = affine_fit(np.array(pv))

	xs,ys,zs = symbols('xs,ys,zs')
	P = np.array([xs,ys,zs])
	plane_func = np.dot(vpv,P - ctd_pv)
	zplane = solve(plane_func,zs)


	return vpv,ctd_pv



