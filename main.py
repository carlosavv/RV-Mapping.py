import numpy as np 
import matplotlib.pyplot as plt 
from LV_longAxis import compute_lvLA
from generateLVpts import genPts
from transform import transform
from plot_LV_RV import plot_lv_rv

def main():

	####### first work with LV to establish a reference for the RV #######

	# load the lv 
	lv = np.loadtxt('Normal2_P0_LV.dat')

	# find points within the vicinity of the maximum z point value
	P = genPts(lv)
	# https://mplcursors.readthedocs.io/en/stable/ 
	# this might be useful in selecting points
	# P = [p0, p1, p2]

	long_axis, mvcenter, apex = compute_lvLA(P, lv)

	# lvData = [long_axis, mvcenter, apex]

	####### Now work with the RV #######
	rv = np.loadtxt('N2_RV_P0.dat')
	plot_lv_rv(lv,rv,mvcenter,apex)

	# Transform the RV into LV centric system 
	print(apex)
	print('')
	print(mvcenter)
	apex, lv, rv = transform(apex,mvcenter,long_axis,lv,rv)
	print(apex)
	
	# transformedData = apex, lv, transformed_rv
	
	mvcenter = np.zeros((3))
	print(mvcenter)

	# Plot the LV-RV system for visualization 
	plot_lv_rv(lv,rv,mvcenter,apex)
	

	# # generates points that fit a plane at the PV
	# pvPt = generatePVpts(rv)
	# # pvPt = [pv_vec, ctd_pv]

	# # generates points that fit a plane at the apex
	# rwPt = generateApexPts(rv,pvPt[1])
	# # rwPt = [rw_vec, ctd_rw]

	# plot_endPlanes(rwPt,pvPt,rv)

	# # Initiate the Bezier Curve (central axis) having fixed end points and normals
	# gp0 = rwPt[1][0]
	# gp1 = pvPt[1][0]
	# ctrl_pts = guess_CA(rwPt, pvPt, gp0, gp1)
	
	# plot_bezier(cpts_0)

	# # section the RV using the CA (central axis)

	# num_sections = 50
	# segment_ctds = section_guess_CA(num_sections,ctrl_pts,rv,rwPt[1],pvPt[1])

	# fig = plt.figure()
	# ax = plt.axes(projection = '3d')
	# ax.scatter(segment_ctds[:,0],segment_ctds[:,1],segment_ctds[:,2])

	# for j in range(0,5):
	# 	ctrl_pts.append(eval_CA(segment_ctds,rwPt[0],pvPt[0]))
	# 	segment_ctds = resection_CA(num_sections, ctrl_pts,rv)

	# plot_bezier(ctrl_pts)

	# remappedRV = remapping_CA(ctrl_pts,rv)
	# straight_axis = np.array([np.zeros((len(clen))),np.zeros((len(clen))),clen])

	# fig = plt.figure()
	# ax = plt.axes(projection = '3d')
	# ax.scatter(remappedRV[:,0],remappedRV[:,1],remappedRV[:,2])
	# ax.plot(straight_axis[:,0],straight_axis[:,1],straight_axis[:,2])


main()