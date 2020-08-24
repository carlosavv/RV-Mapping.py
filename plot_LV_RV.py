import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

def plot_lv_rv(lv,rv,mvcenter,apex):
    
    fig = plt.figure(dpi = 150)
    ax = plt.axes(projection = '3d')
    ax.scatter(lv[:,0],lv[:,1],lv[:,2])
    ax.scatter(rv[:,0],rv[:,1],rv[:,2])
    ax.plot([mvcenter[0],apex[0]],[mvcenter[1],apex[1]],[mvcenter[2],apex[2]],color = 'r', linewidth = 5)

    fig = plt.figure(dpi = 150)
    ax = plt.axes(projection = '3d')
    ax.scatter(rv[:,0],rv[:,1],rv[:,2])
    plt.show()
