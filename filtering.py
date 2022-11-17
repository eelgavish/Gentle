import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
"""
data = pd.read_csv("IMUdata/DrMeryskin1667249917.csv")

time = data['Time'].to_numpy()

acc1 = data[['accX1','accY1','accZ1']].values

N = 3
Wn = 0.1
b,a = scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
zi = scipy.signal.lfilter_zi(b,a)
z, _ = scipy.signal.lfilter(b,a,acc1[:,0],zi=zi*acc1[0,0])
z2, _ = scipy.signal.lfilter(b,a,z,zi=zi*z[0])
y = scipy.signal.filtfilt(b, a, acc1[:,0])

plt.figure
plt.plot(time,acc1[:,0], 'b')
plt.plot(time,z,'r--',time,z2,'r',time,y,'k')
plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice', 'filtfilt'), loc='best')
plt.grid(True)
plt.show()
"""

def butter_filter(N,Wn,input):
	b,a = scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
	y = scipy.signal.filtfilt(b, a, input)
	return y