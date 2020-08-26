import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import mplcursors
import plotly.graph_objects as go 


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
	# mplcursors.cursor(points)

	# plt.show()
	# plt.pause(0.001)
	# fig.canvas.mpl_connect('pick_event', onpick3)