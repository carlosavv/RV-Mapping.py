import numpy as np 
from LV_longAxis import affine_fit
from plot_LV_RV import plot_rv


# generates points that fit a plane at the apex

def genApexPts(rv,pv_ctd):

	# plot_rv(rv)
	# print('Enter x,y,z coordinates located at the free wall: ')
	
	n = 3
	# for now keep this input for testing of code 
	candidate_pt = [-32,-55,-65]
	
	# for i in range(0,n):
		# candidate_pt.append(float(input()))

	vrw = np.subtract(candidate_pt,pv_ctd)
	vrw = vrw/np.linalg.norm(vrw)

	rw_ctd = candidate_pt
	tol = 5 
	rwpts = []

	for i in range(0,len(rv)):
		res = np.dot(vrw,rv[i]-rw_ctd)
		if abs(res) <= tol:
			rwpts.append([rv[i,0],rv[i,1],rv[i,2]])

	rwpts = np.array(rwpts)

	[vrw,foo,rw_ctd] = affine_fit(rwpts)

	vrw = np.array(vrw)
	# print('\n')
	# print("vpv = ",vpv)
	# print("ctd_pv = ", ctd_pv)

	return vrw,rw_ctd
