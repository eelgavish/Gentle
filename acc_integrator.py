import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rotmat as rm
from filtering import butter_filter
import csv

#data = pd.read_csv("IMUdata/DrMeryskin1667249917.csv")
data = pd.read_csv("IMUdata/DrVanardosanastomosis1667251523.csv")
# Time domain
time = data['Time'].to_numpy()

length = len(data)

camAlign = 91
accAlign = 524
camFreq = 60
accFreq = 100
camZero = int(accAlign-camAlign*accFreq/camFreq)

# Initial positions and velocities
p1x = np.empty([length])
p1y = np.empty([length])
p1z = np.empty([length])
v1x = np.empty([length])
v1y = np.empty([length])
v1z = np.empty([length])

r1a = pd.Series({0: [np.eye(3)]})
r2a = pd.Series({0: [np.eye(3)]})
#r1a = r1a.append(pd.Series({2: [np.eye(3)]}))
#print(r1a.get(2)[0])

p2x = np.empty([length])
p2y = np.empty([length])
p2z = np.empty([length])
v2x = np.empty([length])
v2y = np.empty([length])
v2z = np.empty([length])

# open the file in the write mode
f = open('IMUdata/Processed_DrVanardosanastomosis1667251523.csv', 'w', newline='')

c = open('IMUdata/Calibration_DrVanardosanastomosis1667251523.csv', 'w', newline='')

# Camera wrist data
#wrist1 = pd.read_csv("CAMdata/dr_mery_skin_left.csv")
#wrist2 = pd.read_csv("CAMdata/dr_mery_skin_right.csv")
wrist1 = pd.read_csv("CAMdata/dr_vernaldos_anastomosis_left.csv")
wrist2 = pd.read_csv("CAMdata/dr_vernaldos_anastomosis_right.csv")

# create the csv writer for plotter
writer = csv.writer(f)

# create the csv writer for pivot calibration
writerc = csv.writer(c)
#cal_range = [240, 380]
cal_range = [975, 1050]
writerc.writerow([4,cal_range[1]-cal_range[0],'calibration-tool'])

writer.writerow(['p1x','p1y','p1z','p2x','p2y','p2z','w1x','w1y','w1z','w2x','w2y','w2z','bx','by','bz'])
writer.writerow([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

bx = np.empty([length])
by = np.empty([length])
bz = np.empty([length])

#tooltip = np.array([-126.43,-2.28,17.75])
#tooltip = np.array([-2.28,-126.43,17.75]) Mery skin 1
tooltip = np.array([-5.18,-27.8,-14.1]) # venardos anastemosis
tooltip = tooltip/100/3.28

acc1 = data[['accX1','accY1','accZ1']].values
N = 3
Wn = 0.5
accX1 = butter_filter(N,Wn,acc1[:,0])
accY1 = butter_filter(N,Wn,acc1[:,1])
accZ1 = butter_filter(N,Wn,acc1[:,2])

acc2 = data[['accX2','accY2','accZ2']].values
accX2 = butter_filter(N,Wn,acc2[:,0])
accY2 = butter_filter(N,Wn,acc2[:,1])
accZ2 = butter_filter(N,Wn,acc2[:,2])

w1x = butter_filter(N,Wn,wrist1['X'].to_numpy())
w2x = butter_filter(N,Wn,wrist2['X'].to_numpy())
w1y = butter_filter(N,Wn,wrist1['Y'].to_numpy())
w2y = butter_filter(N,Wn,wrist2['Y'].to_numpy())
w1z = butter_filter(N,Wn,wrist1['Z'].to_numpy())
w2z = butter_filter(N,Wn,wrist2['Z'].to_numpy())

len1 = len(w1x)
len2 = len(w2x)

# Calculating things
# Jerk
acc1a = np.empty([length])
acc1b = np.empty([length])
acc2a = np.empty([length])
acc2b = np.empty([length])
acc1a[0] = np.linalg.norm(acc1[0])
acc2a[0] = np.linalg.norm(acc2[0])
acc1b[0] = np.linalg.norm(np.array([accX1[0],accY1[0],accZ1[0]]))
acc2b[0] = np.linalg.norm(np.array([accX2[0],accY2[0],accZ2[0]]))
jerk1a = np.empty([length])
jerk2a = np.empty([length])
jerk1b = np.empty([length])
jerk2b = np.empty([length])

# Angular Velocity Variability

# Acceleration Variability
# Deliberate Hand Movements

# Integrate accelerations to velocities and positions
for i in range(0,length-1):
    t = time[i]
    dt = time[i+1] - t
    v1x[i+1] = dt*accX1[i]
    v1y[i+1] = dt*accY1[i]
    v1z[i+1] = dt*accZ1[i]
    p1x[i+1] = dt*v1x[i+1]*1000/3.28
    p1y[i+1] = dt*v1y[i+1]*1000/3.28
    p1z[i+1] = dt*v1z[i+1]*1000/3.28
    v1x[i+1] = dt*accX2[i]*9.81
    v2y[i+1] = dt*accY2[i]*9.81
    v2z[i+1] = dt*accZ2[i]*9.81
    p2x[i+1] = dt*v2x[i+1]*1000/3.28
    p2y[i+1] = dt*v2y[i+1]*1000/3.28
    p2z[i+1] = dt*v2z[i+1]*1000/3.28

    w1 = np.array([data.at[i,'wX1'],data.at[i,'wY1'],data.at[i,'wZ1']])
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
    bx[i+1] = p1x[i+1]+b[0]
    by[i+1] = p1y[i+1]+b[1]
    bz[i+1] = p1z[i+1]+b[2]

    iw = int((i-camZero)*camFreq/accFreq)
    if iw < len1:
        w1xt = w1x[iw]
        w1yt = w1y[iw]
        w1zt = w1z[iw]
    else:
        w1xt = 0
        w1yt = 0
        w1zt = 0        
    if iw < len2:
        w2xt = w2x[iw]
        w2yt = w2y[iw]
        w2zt = w2z[iw]
    else:
        w2xt = 0
        w2yt = 0
        w2zt = 0

    # write a row to the csv file
    writer.writerow([p1x[i+1],p1y[i+1],p1z[i+1],p2x[i+1],p2y[i+1],p2z[i+1],w1xt,w1yt,w1zt,w2xt,w2yt,w2zt,bx[i+1],by[i+1],bz[i+1]])

    if i >= cal_range[0] and i < cal_range[1]:
        writerc.writerow([p1x[i+1],p1y[i+1],p1z[i+1]])
        writerc.writerow([p1x[i+1]+r1[0,0],p1y[i+1]+r1[1,0],p1z[i+1]+r1[2,0]])
        writerc.writerow([p1x[i+1]+r1[1,0],p1y[i+1]+r1[1,1],p1z[i+1]+r1[1,2]])
        writerc.writerow([p1x[i+1]+r1[2,0],p1y[i+1]+r1[2,1],p1z[i+1]+r1[2,2]])

# close the file
f.close()