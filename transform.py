import numpy as np 


def R1(t):
	return np.array([ [np.cos(t), np.sin(t), 0],
					  [-np.sin(t), np.cos(t), 0],
					  [	0, 	0,	1	]	])
def R2(t):
	return np.array([ [np.cos(t), 0, np.sin(t)],
				 	  [0,	1,	0],
				  	  [	-np.sin(t), 0, np.cos(t)]])


def transform(apex,mvcenter,longAxis, lv,rv):
	#### shift origin to center of mitral valve ####

	# project the long axis vector onto the x-y plane 
	rv = rv - np.ones((len(rv),1))*mvcenter
	lv = lv - np.ones((len(lv),1))*mvcenter

	apex = apex - mvcenter


	xy_proj = np.array([[1,0,0],[0,1,0],[0,0,0]])
	longAxis_proj = np.matmul(xy_proj,longAxis) 
	longAxis_proj = longAxis_proj

	angle_x = np.pi+np.arccos(np.dot(np.array([1,0,0]),longAxis_proj)/np.linalg.norm(longAxis_proj))
	angle_z = np.arccos(np.dot(np.array([0,0,1]),longAxis)/np.linalg.norm(longAxis))

	# transform the data 
	lv_temp = []
	transformed_lv = []
	for i in range(0,len(lv)):
		lv_temp.append(np.matmul(R1(angle_x),lv[i]))
		transformed_lv.append(np.matmul(R2(angle_z),lv_temp[i]))

	transformed_lv = np.array(transformed_lv)
	rv_temp = []
	transformed_rv = []
	for i in range(0,len(rv)):
		rv_temp.append(np.matmul(R1(angle_x),rv[i]))
		transformed_rv.append(np.matmul(R2(angle_z),rv_temp[i]))

	transformed_rv = np.array(transformed_rv)
	apex = np.matmul(np.matmul(R2(angle_z),R1(angle_x)),apex)

	return apex, transformed_lv, transformed_rv
