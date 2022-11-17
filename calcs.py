import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import rotmat as rm
from filtering import butter_filter

#data = pd.read_csv("IMUdata/DrMeryskin1667249917.csv")
data = pd.read_csv("IMUdata/DrVanardosanastomosis1667251523.csv")
fing1 = pd.read_csv("CAMdata/dr_vernaldos_anastomosis_left.csv")
fing2 = pd.read_csv("CAMdata/dr_vernaldos_anastomosis_right.csv")
# Time domain
time = data['Time'].to_numpy()

length = len(data)

camRnge = [6319,6919]
camZero = -220
rnge = [int(camRnge[0]*100/60),int(camRnge[1]*100/60)]
#rnge = [4522, 5522]
print("Cam Indices: "+str(rnge[0])+" : "+str(rnge[1]))
print("Acc Indices: "+str(camRnge[0])+" : "+str(camRnge[1]))
lngth = rnge[1]-rnge[0]
print("Time: "+str(time[rnge[1]]-time[rnge[0]]))

acc1 = data[['accX1','accY1','accZ1']].values
acc2 = data[['accX2','accY2','accZ2']].values*9.81

# subtract out gravity via orientation
# end orientation
acc1x = np.mean(acc1[500:600,0])
acc1y = np.mean(acc1[500:600,1])
acc1z = np.mean(acc1[500:600,2])
acc1t = np.array([acc1x,acc1y,acc1z])

g = [0,0,-9.8]
Ra = rm.align(acc1t,g)
dt = time[1] - time[0]

w1x = np.flip(data['wX1'].values)
w1y = np.flip(data['wY1'].values)
w1z = np.flip(data['wZ1'].values)
for i in range(0,length):
    w1 = np.array([w1x[i],w1y[i],w1z[i]])
    th1 = np.linalg.norm(w1)*dt*np.pi/180
    if th1 != 0:
        w1 = w1/th1
        r1 = rm.AARotMat(w1,th1)
    else:
        r1 = np.eye(3)
    #r1a = r1a.append(pd.Series({i+1: [np.dot(r1a.get(i)[0],r1)]}))

    Ra = np.dot(Ra,r1)
    acc1[length-1-i] = np.dot(Ra,acc1[length-1-i])-g

# filter data
N = 3
Wn = 0.4
accX1 = butter_filter(N,Wn,acc1[:,0])
accY1 = butter_filter(N,Wn,acc1[:,1])
accZ1 = butter_filter(N,Wn,acc1[:,2])
accX2 = butter_filter(N,Wn,acc2[:,0])
accY2 = butter_filter(N,Wn,acc2[:,1])
accZ2 = butter_filter(N,Wn,acc2[:,2])

velX1 = scipy.integrate.cumtrapz(accX1,dx=dt)
velY1 = scipy.integrate.cumtrapz(accY1,dx=dt)
velZ1 = scipy.integrate.cumtrapz(accZ1,dx=dt)
velX2 = scipy.integrate.cumtrapz(accX2,dx=dt)
velY2 = scipy.integrate.cumtrapz(accY2,dx=dt)
velZ2 = scipy.integrate.cumtrapz(accZ2,dx=dt)

posX1 = scipy.integrate.cumtrapz(velX1,dx=dt)
posY1 = scipy.integrate.cumtrapz(velY1,dx=dt)
posZ1 = scipy.integrate.cumtrapz(velZ1,dx=dt)
posX2 = scipy.integrate.cumtrapz(velX2,dx=dt)
posY2 = scipy.integrate.cumtrapz(velY2,dx=dt)
posZ2 = scipy.integrate.cumtrapz(velZ2,dx=dt)

# Jerk
jerkX1 = np.gradient(accX1)
jerkY1 = np.gradient(accY1)
jerkZ1 = np.gradient(accZ1)
jerkX2 = np.gradient(accX2)
jerkY2 = np.gradient(accY2)
jerkZ2 = np.gradient(accZ2)
jerk1 = np.sqrt(jerkX1**2+jerkY1**2+jerkZ1**2)
jerk2 = np.sqrt(jerkX2**2+jerkY2**2+jerkZ2**2)

# Camera Jerk Calcs
N = 3
Wn = 0.6
camX1 = butter_filter(N,Wn,fing1['X'].values)
camY1 = butter_filter(N,Wn,fing1['Y'].values)
camZ1 = butter_filter(N,Wn,fing1['Z'].values)
camX2 = butter_filter(N,Wn,fing2['X'].values)
camY2 = butter_filter(N,Wn,fing2['Y'].values)
camZ2 = butter_filter(N,Wn,fing2['Z'].values)

camVelX1 = np.gradient(camX1)
camVelY1 = np.gradient(camY1)
camVelZ1 = np.gradient(camZ1)
camVelX2 = np.gradient(camX2)
camVelY2 = np.gradient(camY2)
camVelZ2 = np.gradient(camZ2)

camAccX1 = np.gradient(camVelX1)
camAccY1 = np.gradient(camVelY1)
camAccZ1 = np.gradient(camVelZ1)
camAccX2 = np.gradient(camVelX2)
camAccY2 = np.gradient(camVelY2)
camAccZ2 = np.gradient(camVelZ2)
camAcc1 = np.sqrt(camAccX1**2+camAccY1**2+camAccZ1**2)
camAcc2 = np.sqrt(camAccX2**2+camAccY2**2+camAccZ2**2)

camJerkX1 = np.gradient(camAccX1)
camJerkY1 = np.gradient(camAccY1)
camJerkZ1 = np.gradient(camAccZ1)
camJerkX2 = np.gradient(camAccX2)
camJerkY2 = np.gradient(camAccY2)
camJerkZ2 = np.gradient(camAccZ2)
camJerk1 = np.sqrt(camJerkX1**2+camJerkY1**2+camJerkZ1**2)
camJerk2 = np.sqrt(camJerkX2**2+camJerkY2**2+camJerkZ2**2)

#acc1a = np.empty([lngth])
acc1b = np.empty([lngth])
#acc2a = np.empty([lngth])
acc2b = np.empty([lngth])

#acc1a[0] = np.linalg.norm(acc1[0])
#acc2a[0] = np.linalg.norm(acc2[0])
acc1b[0] = np.linalg.norm(np.array([accX1[0],accY1[0],accZ1[0]]))
acc2b[0] = np.linalg.norm(np.array([accX2[0],accY2[0],accZ2[0]]))

# Angular Velocity Variability
data['norm'] = data.apply(lambda row : np.linalg.norm([row['wX1'],row['wY1'],row['wZ1']]),axis=1)
norm = data['norm'].values[rnge[0]:rnge[1]]
wvar = norm.mean()
print("Angular Velocity Variability 1 Mean: "+str(wvar))
wdev = norm.std(ddof=0)
print("Angular Velocity Variability 1 Stdev: "+str(wdev))

data['norm'] = data.apply(lambda row : np.linalg.norm([row['wX2'],row['wY2'],row['wZ2']]),axis=1)
norm = data['norm'].values[rnge[0]:rnge[1]]*np.pi/180
wvar = norm.mean()
print("Angular Velocity Variability 2 Mean: "+str(wvar))
wdev = norm.std(ddof=0)
print("Angular Velocity Variability 2 Stdev: "+str(wdev))

# Acceleration Variability
data['norm'] = data.apply(lambda row : np.linalg.norm([row['accX1'],row['accY1'],row['accZ1']]),axis=1)
norm = data['norm'].values[rnge[0]:rnge[1]]
wvar = norm.mean()
print("Acceleration Variability 1 Mean: "+str(wvar))
wdev = norm.std(ddof=0)
print("Acceleration Variability 1 Stdev: "+str(wdev))

data['norm'] = data.apply(lambda row : np.linalg.norm([row['accX2'],row['accY2'],row['accZ2']]),axis=1)
norm = data['norm'].values[rnge[0]:rnge[1]]
wvar = norm.mean()
print("Acceleration Variability 2 Mean: "+str(wvar))
wdev = norm.std(ddof=0)
print("Acceleration Variability 2 Stdev: "+str(wdev))

# Acceleration Variability Camera Data
wvar = camAcc1[camRnge[0]:camRnge[1]].mean()
print("Acceleration Variability Cam 1 Mean: "+str(wvar))
wdev = camAcc1[camRnge[0]:camRnge[1]].std(ddof=0)
print("Acceleration Variability Cam 1 Stdev: "+str(wdev))

wvar = camAcc2[camRnge[0]:camRnge[1]].mean()
print("Acceleration Variability Cam 2 Mean: "+str(wvar))
wdev = camAcc2[camRnge[0]:camRnge[1]].std(ddof=0)
print("Acceleration Variability Cam 2 Stdev: "+str(wdev))

# Deliberate Hand Movements
# Count the # times accel > 10
count1 = 0
boo1 = False
count2 = 0
boo2 = False

# Path Length
pth1 = 0
pth2 = 0
pth1c = 0
pth2c = 0
# Initial positions and velocities
p1x = np.empty([lngth])
p1y = np.empty([lngth])
p1z = np.empty([lngth])
v1x = np.empty([lngth])
v1y = np.empty([lngth])
v1z = np.empty([lngth])
p2x = np.empty([lngth])
p2y = np.empty([lngth])
p2z = np.empty([lngth])
v2x = np.empty([lngth])
v2y = np.empty([lngth])
v2z = np.empty([lngth])


j = 0
for i in range(rnge[0],rnge[1]-1):
    t = time[i]
    dt = time[i+1] - t

    # Calculating things
    # Jerk
    #acc1a[i+1] = np.linalg.norm(acc1[i+1])
    #acc2a[i+1] = np.linalg.norm(acc2[i+1])
    acc1b[j+1] = np.linalg.norm(np.array([accX1[i],accY1[i],accZ1[i]]))
    acc2b[j+1] = np.linalg.norm(np.array([accX2[i],accY2[i],accZ2[i]]))

    #jerk1a[i+1] = abs(dt*(acc1a[i+1]-acc1a[i]))
    #jerk1b[j] = jerk1[i]
    #jerk2a[i+1] = abs(dt*(acc2a[i+1]-acc2a[i]))
    #jerk2b[j] = jerk2[i]

	# To find Path length
    #pth1 = pth1 + abs(np.linalg.norm(np.array([p1x[j+1]-p1x[j],p1y[j+1]-p1y[j],p1z[j+1]-p1z[j]])))
    #pth2 = pth2 + abs(np.linalg.norm(np.array([p2x[j+1]-p2x[j],p2y[j+1]-p2y[j],p2z[j+1]-p2z[j]])))
    pth1 = pth1 + np.linalg.norm(np.array([posX1[i+1]-posX1[i],posY1[i+1]-posY1[i],posZ1[i+1]-posZ1[i]]))
    pth2 = pth2 + np.linalg.norm(np.array([posX2[i+1]-posX2[i],posY2[i+1]-posY2[i],posZ2[i+1]-posZ2[i]]))

    # Deliberate Hand Movements
    if acc1b[j] >= 10 and not boo1:
            count1 = count1+1
            boo1 = True
    elif acc1b[j] < 10:
        boo1 = False
	
    if acc2b[j] >= 10 and not boo1:
            count2 = count2+1
            boo2 = True
    elif acc2b[j] < 10:
        boo2 = False

    j = j+1

#for j in range(camRnge[0],camRnge[1]-1):
    #pth1c = pth1c + np.linalg.norm(np.array([camX1[j+1]-camX1[j],camY1[j+1]-camY1[j],camZ1[j+1]-camZ1[j]]))
    #pth2c = pth2c + np.linalg.norm(np.array([camX2[j+1]-camX2[j],camY2[j+1]-camY2[j],camZ2[j+1]-camZ2[j]]))

print("Path length 1: "+str(pth1))
print("Path length 2: "+str(pth2))
print("Path length 1 Cam: "+str(pth1c))
print("Path length 2 Cam: "+str(pth2c))

print("Deliberate count 1: "+str(count1))
print("Deliberate count 2: "+str(count2))

# Mean Jerk
mjerk1 = np.mean(jerk1[rnge[0]:rnge[1]])
print("Mean Jerk 1: "+str(mjerk1))
mjerk2 = np.mean(jerk2[rnge[0]:rnge[1]])
print("Mean Jerk 2: "+str(mjerk2))
mjerk1c = np.mean(camJerk1[camRnge[0]:camRnge[1]])
print("Mean Jerk 1 Cam: "+str(mjerk1c))
mjerk2c = np.mean(camJerk2[camRnge[0]:camRnge[1]])
print("Mean Jerk 2 Cam: "+str(mjerk2c))

plt.figure(1)
timep = time[0:len(posX1)]
timev = time[0:len(velX1)]
timea = time[0:len(accX1)]
plt.subplot(3,3,1)
plt.plot(timep,posX1)
plt.subplot(3,3,4)
plt.plot(timev,velX1)
plt.subplot(3,3,7)
plt.plot(timea,accX1)
plt.subplot(3,3,2)
plt.plot(timep,posY1)
plt.subplot(3,3,5)
plt.plot(timev,velY1)
plt.subplot(3,3,8)
plt.plot(timea,accY1)
plt.subplot(3,3,3)
plt.plot(timep,posZ1)
plt.subplot(3,3,6)
plt.plot(timev,velZ1)
plt.subplot(3,3,9)
plt.plot(timea,accZ1)

time = time[rnge[0]:rnge[1]]
cutoff = np.ones(len(time))*10
plt.figure(2)
plt.subplot(2,2,1)
#plt.plot(time,acc1a,'r',time,acc1b,'r--')
#plt.legend(('acc 1', 'acc 1 filtered'), loc='best')
plt.plot(time,acc1b,'r',time,cutoff,'r--')
plt.legend(('acc 1', 'Deliberate Threshold'), loc='best')
plt.subplot(2,2,2)
#plt.plot(time,acc1a,'b',time,acc1b,'k')
#plt.legend(('acc 2', 'acc 2 filtered'), loc='best')
plt.plot(time,acc2b,'b',time,cutoff,'k')
plt.legend(('acc 2', 'Deliberate Threshold'), loc='best')

mjerk1 = np.ones(len(time))*mjerk1
mjerk2 = np.ones(len(time))*mjerk2
plt.subplot(2,2,3)
plt.plot(time,jerk1[rnge[0]:rnge[1]],'r',time,mjerk1,'r--')
plt.legend(('jerk 1', 'mean jerk'), loc='best')
plt.subplot(2,2,4)
plt.plot(time,jerk2[rnge[0]:rnge[1]],'b',time,mjerk2,'k')
plt.legend(('jerk 2', 'mearn jerk'), loc='best')
plt.show()