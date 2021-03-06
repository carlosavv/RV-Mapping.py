import numpy as np 
from tools import affine_fit
from plot_LV_RV import plot_rv



# generates points that fit a plane at the PV
# contains a point lying in the plane that fits the PV -> ctd_pv
# contains normal of the plane fitting the PV -> vpv 

def genPVpts(rv):

	print('Enter x,y,z coordinates located at the PV: ')
	plot_rv(rv)
	n = 3
	# for now keep this input for testing of code 
	# candidate_pt = [2,-34,17]
	candidate_pt = []
	
	for i in range(0,n):
		candidate_pt.append(float(input()))


	lowerBound = np.array(rv[rv[:,2] == min(rv[:,2])][0])
	vpv = np.subtract(candidate_pt,lowerBound)
	vpv = vpv/np.linalg.norm(vpv)


	ctd_pv = candidate_pt

	tol = 5 
	pv = []
	for i in range(0,len(rv)):
		res = np.dot(vpv,rv[i]-ctd_pv)
		if abs(res) <= tol:
			pv.append([rv[i,0],rv[i,1],rv[i,2]])

	pv = np.array(pv)

	[foo,vpv,ctd_pv] = affine_fit(pv)

	vpv = np.array(vpv)
	# print('\n')
	# print("vpv = ",vpv)
	# print("ctd_pv = ", ctd_pv)

	return vpv,ctd_pv
