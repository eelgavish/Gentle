import random
import matplotlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import rotmat as rm

data = pd.read_csv("IMUdata/DrVanardosanastomosis1667251523.csv")
processed_data = pd.read_csv('IMUdata/Processed_DrVanardosanastomosis1667251523.csv')

# Time domain
time = data['Time'].to_numpy()

# Data range to plot
rnge = [4580,5580]

length = rnge[1]-rnge[0]

#tooltip = np.array([-126.43,-2.28,17.75])
#tooltip = np.array([-2.28,-126.43,17.75]) Mery skin 1
tooltip = np.array([200,0,0]) # venardos anastemosis
tooltip = tooltip/100/3.28

# Initial positions and velocities
p1x = np.empty([length])
p1y = np.empty([length])
p1z = np.empty([length])
v1x = np.empty([length])
v1y = np.empty([length])
v1z = np.empty([length])

r1a = pd.Series({0: [np.eye(3)]})
r2a = pd.Series({0: [np.eye(3)]})

p2x = np.empty([length])
p2y = np.empty([length])
p2z = np.empty([length])
v2x = np.empty([length])
v2y = np.empty([length])
v2z = np.empty([length])

w1x = np.empty([length])
w1y = np.empty([length])
w1z = np.empty([length])
w2x = np.empty([length])
w2y = np.empty([length])
w2z = np.empty([length])

bx = np.empty([length])
by = np.empty([length])
bz = np.empty([length])

# Integrate accelerations to velocities and positions
for i in range(0,length-1):
    t = time[i]
    dt = time[i+1] - t

    p1x[i] = processed_data.at[rnge[0]+i,'p1x']
    p1y[i] = processed_data.at[rnge[0]+i,'p1y']
    p1z[i] = processed_data.at[rnge[0]+i,'p1z']
    p2x[i] = processed_data.at[rnge[0]+i,'p2x']
    p2y[i] = processed_data.at[rnge[0]+i,'p2y']
    p2z[i] = processed_data.at[rnge[0]+i,'p2z']

    w1x[i] = processed_data.at[rnge[0]+i,'w1x']
    w1y[i] = processed_data.at[rnge[0]+i,'w1y']
    w1z[i] = processed_data.at[rnge[0]+i,'w1z']
    w2x[i] = processed_data.at[rnge[0]+i,'w2x']
    w2y[i] = processed_data.at[rnge[0]+i,'w2y']
    w2z[i] = processed_data.at[rnge[0]+i,'w2z']
    
    #bx[i] = processed_data.at[rnge[0]+i,'bx']
    #by[i] = processed_data.at[rnge[0]+i,'by']
    #bz[i] = processed_data.at[rnge[0]+i,'bz']
 
    w1 = np.array([data.at[rnge[0]+i,'wX1'],data.at[rnge[0]+i,'wY1'],data.at[rnge[0]+i,'wZ1']])
    th1 = np.linalg.norm(w1)*dt 
    if th1 != 0:
        w1 = w1/th1
        r1 = rm.AARotMat(w1,th1)
    else:
        r1 = np.eye(3)
    r1a = r1a.append(pd.Series({i+1: [np.dot(r1a.get(i)[0],r1)]}))

    w2 = np.array([data.at[i,'wX2'],data.at[i,'wY2'],data.at[i,'wZ2']])
    w2 = w2*np.pi/180
    th2 = np.linalg.norm(w2)*dt
    if th2 != 0:
        w2 = w2/th2
        r2 = rm.AARotMat(w2,th2)
    else:
        r2 = np.eye(3)
    r2a = r2a.append(pd.Series({i+1: [np.dot(r2a.get(i)[0],r2)]}))

    b = np.dot(r1a.get(i)[0],tooltip)
    bx[i] = p1x[i]+b[0]
    by[i] = p1y[i]+b[1]
    bz[i] = p1z[i]+b[2]

# Compression ratio
c = 1
p1x = np.mean(p1x.reshape(-1,c), axis=1)
p1y = np.mean(p1y.reshape(-1,c), axis=1)
p1z = np.mean(p1z.reshape(-1,c), axis=1)
p2x = np.mean(p2x.reshape(-1,c), axis=1)
p2y = np.mean(p2y.reshape(-1,c), axis=1)
p2z = np.mean(p2z.reshape(-1,c), axis=1)
w1x = np.mean(w1x.reshape(-1,c), axis=1)
w1y = np.mean(w1y.reshape(-1,c), axis=1)
w1z = np.mean(w1z.reshape(-1,c), axis=1)
w2x = np.mean(w2x.reshape(-1,c), axis=1)
w2y = np.mean(w2y.reshape(-1,c), axis=1)
w2z = np.mean(w2z.reshape(-1,c), axis=1)
bx = np.mean(bx.reshape(-1,c), axis=1)
by = np.mean(by.reshape(-1,c), axis=1)
bz = np.mean(bz.reshape(-1,c), axis=1)
length = length/c

matplotlib.use('TkAgg')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
axrange = [-0.4,0.4]
sc = 0.1 # axes scaling

mngr = plt.get_current_fig_manager()
window = mngr.window


backTrack = 500
def update():
    n=n_slider.get()
    ax.clear()
    # Plot Position Data
    ax.scatter(p1x[n],p1y[n],p1z[n],c='r') # Tooltip Acc
    ax.scatter(p2x[n],p2y[n],p2z[n],c='b') # Arm Acc
    # Plot Backtrack
    ax.plot(p1x[n-backTrack:n],p1y[n-backTrack:n],p1z[n-backTrack:n],c='k')
    ax.plot(p2x[n-backTrack:n],p2y[n-backTrack:n],p2z[n-backTrack:n],c='b')
    # Plot Frame Axes
    r1b = r1a.get(n)[0]
    ax.plot([p1x[n],p1x[n]+r1b[0,0]*sc],[p1y[n],p1y[n]+r1b[1,0]*sc],[p1z[n],p1z[n]+r1b[2,0]*sc],c='r') # Rotated Coordinate Frame Axes
    ax.plot([p1x[n],p1x[n]+r1b[0,1]*sc],[p1y[n],p1y[n]+r1b[1,1]*sc],[p1z[n],p1z[n]+r1b[2,1]*sc],c='g')
    ax.plot([p1x[n],p1x[n]+r1b[0,2]*sc],[p1y[n],p1y[n]+r1b[1,2]*sc],[p1z[n],p1z[n]+r1b[2,2]*sc],c='b')
    r2b = r2a.get(n)[0]
    ax.plot([p2x[n],p2x[n]+r2b[0,0]*sc],[p2y[n],p2y[n]+r2b[1,0]*sc],[p2z[n],p2z[n]+r2b[2,0]*sc],c='r') # Rotated Coordinate Frame Axes
    ax.plot([p2x[n],p2x[n]+r2b[0,1]*sc],[p2y[n],p2y[n]+r2b[1,1]*sc],[p2z[n],p2z[n]+r2b[2,1]*sc],c='g')
    ax.plot([p2x[n],p2x[n]+r2b[0,2]*sc],[p2y[n],p2y[n]+r2b[1,2]*sc],[p2z[n],p2z[n]+r2b[2,2]*sc],c='b')
    # Plot wrist
    # Frequency Transform
    #ax.scatter(w1x[n],w1y[n],w1z[n],c='k')
    #ax.plot(w1x[n-backTrack:n],w1y[n-backTrack:n],w1z[n-backTrack:n],c='m')
    #ax.scatter(w2x[n],w2y[n],w2z[n],c='k')
    #ax.plot(w2x[n-backTrack:n],w2y[n-backTrack:n],w2z[n-backTrack:n],c='m')
    # Plot tooltip
    #ax.scatter(bx[n],by[n],bz[n],c='m')
    #ax.plot([p1x[n],bx[n]],[p1y[n],by[n]],[p1z[n],bz[n]],c='m')
    #ax.plot([bx[n],bx[n]+r1b[0,0]*sc],[by[n],by[n]+r1b[1,0]*sc],[bz[n],bz[n]+r1b[2,0]*sc],c='r') # Rotated Coordinate Frame Axes
    #ax.plot([bx[n],bx[n]+r1b[0,1]*sc],[by[n],by[n]+r1b[1,1]*sc],[bz[n],bz[n]+r1b[2,1]*sc],c='g')
    #ax.plot([bx[n],bx[n]+r1b[0,2]*sc],[by[n],by[n]+r1b[1,2]*sc],[bz[n],bz[n]+r1b[2,2]*sc],c='b')
    #ax.plot(bx[n-backTrack:n],by[n-backTrack:n],bz[n-backTrack:n],c='m')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(axrange)
    ax.set_ylim(axrange)
    ax.set_zlim(axrange)
    fig.canvas.draw_idle()

n_slider = tk.Scale(master=window,variable=tk.IntVar(), from_=backTrack, to=length-1, label="Time Steps "+str(rnge[0])+" to "+str(rnge[1]-1), orient=tk.HORIZONTAL,length=int(fig.bbox.width), width=int(fig.bbox.height * 0.05), command = lambda i: update())
n_slider.set(backTrack+1)
n_slider.pack()

def close():
    plt.close('all') 
    window.quit()

button = tk.Button(master=window, text="Quit", command=close)

button.pack(side=tk.BOTTOM)

fig.show()
tk.mainloop()