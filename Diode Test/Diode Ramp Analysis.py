# -*- coding: utf-8 -*-
"""
Diode Ramp Analysis v2.0

Lukas Kostal, 14.8.2023, PSI
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as ss
import sys


# function to read arguments from console
def garg(*args):
    args = list(args)
    arg_sys = sys.argv
    
    for i in range(1, len(arg_sys)):
        args[i-1] = type(args[i-1])(arg_sys[i])
        
    return args


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)


# function to apply low pass Butterworth filter
def butter_lp(arr, fs, fc, order):
    w = fc / fs * 2
    b, a = ss.butter(order, Wn=w, btype='low', analog=False)
    arr = ss.filtfilt(b, a, arr)
    return arr


# apply a low pass butterworth filter
LPF = True

# sampling frequency in Hz
fs = 10
# cutoff frequency in Hz
fc = 1

# data to be analysed
data = 'Diode_081416_D1_D2_Ru'

# array of calibration constants for channels in uWV^-1
cal_arr = np.array([2670, 3540, 2817, 4890])

# specify or input filename of data to be analyzed
data, LPF, fc = garg(data, LPF, fc)

# color for plotting
colr = ['royalblue', 'orange', 'limegreen']

# load the data
readout = np.genfromtxt(f'Data_renamed/{data}.txt', unpack=True, delimiter=',', skip_footer=1)

# set all -ve measurements to 0 and discard the first and last 3 measurements
V_arr = readout[:3, 3:-3]
V_arr[V_arr <= 0] = 0

# discrad firs and last 3 TA current measurements and convert to mA
Iact = -readout[3, 3:-3] * cal_arr[3]

# get the no of measurements
n = len(V_arr[0, :])

# array of sampling times in seconds and in hours
ts = np.arange(0, n) / 10
th = ts / 3600

# apply low pass butterworth filter
if LPF == True:
    for i in range(0, 3):
        V_arr[i, :] = butter_lp(V_arr[i, :], fs, fc, 1)

# apply calibration cosntants to get power
P_arr = np.copy(V_arr)
for i in range(0, 3):
    P_arr[i, :] *= cal_arr[i]
    
# calculate rate of change of power wrt TA current
with np.errstate(divide='ignore'):
    dP_arr = np.diff(P_arr) / np.diff(Iact)

# get total power by adding power at Ch1 and Ch2
P_tot = P_arr[0, :] + P_arr[1, :]

# calcualte ratio of Ch1 to Ch2 and Ch2 to Ch3
R12 = P_arr[0, :] / P_arr[1, :]
R23 = P_arr[1, :] / P_arr[2, :]

# parameters for plotting measured power against TA current
plt.figure(1)
plt.title(f'Measured Power \n Dataset: {data}', pad=40)
plt.xlabel('TA current $I_{act}$ (mA)')
plt.ylabel(r'power $P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, 3):
    plt.plot(Iact, P_arr[i, :], c=colr[i], label=f'Ch{i+1}')
plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_power.png', dpi=300, bbox_inches='tight')

# parameters for plotting dP/dI agaginst TA current
plt.figure(2)
plt.title(f'Rate of Change of Power wrt TA Current \n Dataset: {data}', pad=40)
plt.xlabel('TA current $I_{act}$ (mA)')
plt.ylabel(r'$dP / I_{act}$ ($\mu W mA^{-1}$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, 3):
    plt.plot(Iact[:-1], dP_arr[i, :], c=colr[i], label=f'Ch{i+1}')
plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_rate.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch1 to Ch2
plt.figure(3)
plt.title(f'Measured Beamsplitter Ratio Ch1 to Ch2 \n Dataset: {data}')
plt.xlabel('TA current $I_{act}$ (mA)')
plt.ylabel(r'ratio $R_{12} = Ch1 / Ch2$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(Iact, R12, c=colr[0])
plt.savefig(f'Output/{data}_ratio12.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch1 to Ch2
plt.figure(4)
plt.title(f'Measured Ratio of Ch2 to Ch3 \n Dataset: {data}')
plt.xlabel('TA current $I_{act}$ (mA)')
plt.ylabel(r'ratio $R_{23} = Ch2 / Ch3$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(Iact, R23, c=colr[0])
plt.savefig(f'Output/{data}_ratio23.png', dpi=300, bbox_inches='tight')

plt.show()
        
    


