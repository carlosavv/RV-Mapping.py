import numpy as np 

#  

def genPVpts(rv):

	idxz = rv[rv[:,3] == min(rv[:,3])]

	candidate_pt = input("Enter a point [x,y,z] located at the PV: ")

	vpv = candidate_pt - rv[idxz]
	vpv = vpv/np.linalg.norm(vpv)

	ctd_pv = candidate_pt

	tol = 5 
	pv = []
	for i in range(0,len(rv)):
		res = np.dot(vpv,rv[i]-ctd_pv)

		if abs(res) <= tol:

