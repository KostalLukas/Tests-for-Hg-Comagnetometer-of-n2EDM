# -*- coding: utf-8 -*-
"""
Diode Comparison Analysis v1.1

Lukas Kostal, 6.9.2023, PSI
"""

import numpy as np
from matplotlib import pyplot as plt
import glob as gb
import sys


# ignore all warnings from numpy
np.seterr(all="ignore")


# function to read arguments from console
def garg(*args):
    args = list(args)
    arg_sys = sys.argv
    for i in range(1, len(arg_sys)):
        if type(args[i-1]) == bool:
            args[i-1] = arg_sys[i] == 'True'
        else:
            args[i-1] = type(args[i-1])(arg_sys[i])
    return args

    
# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)
        

# power of laser during test in uW and corresponding error
P = 1680
P_err = 0.05

# channel voltage range setting on DAQ
ch_set = 2

# ampling frequency in Hz
fs = 10
        
P = garg(P)[0]

# get a sorted list of all datasets in the Data directory
ds = gb.glob('Data/Diode_D**.txt')
ds.sort()

n_pds = len(ds)

# map of DAQ channel voltage ranges in V
ch_map = np.array([10, 5, 2.5, 1.25])

# empty arrays to hold results for each diode
pds = np.zeros(n_pds)
V_avg = np.zeros(n_pds)
V_std = np.zeros(n_pds)
V_sem = np.zeros(n_pds)
V_ptp = np.zeros(n_pds)

# parameters for plotting measured power over time in the loop
plt.figure(1)
plt.title('DAQ Input Voltage over Time', pad=40)
plt.xlabel('time $t$ (s)')
plt.ylabel('voltage $V$ (V)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()
    
for i in range(0, n_pds):
    pds[i] = int(ds[i][-5])
    
    # load the data for the given photodiode
    V = np.genfromtxt(ds[i], usecols=(0), unpack=True, delimiter=',')
    
    # set all -ve measurements equal to 0
    V[V <= 0] = 0
    
    # discrad fist and last 3 measurements
    V = V[3:-3]
    V[V <= 0] = 0
    
    # get no of measurements for given photodiode
    n = len(V)
    
    # if test lasted more than 150s crop it at 110s
    if n > 150:
        V = V[0: 105]
        n = 105
        
    # array of sampling times
    t = np.arange(n) / fs
    
    # calculate mean, standard deviation and standard error of measured voltage
    V_avg[i] = np.mean(V)
    V_std[i] = np.std(V)
    V_sem[i] = V_std[i] / np.sqrt(n)
    V_ptp[i] = np.ptp(V)
    
    plt.plot(t, V, label=f'D{pds[i]:.0f}')

plt.legend(loc=(-0.2, 1.05), ncol=n_pds)
plt.savefig('Output/Comparison_pds.png', dpi=300, bbox_inches='tight')

cal_arr = P / (V_avg * ch_map[ch_set-2])
cal_err = cal_arr * np.sqrt(P_err**2 + V_sem**2)
    
# print the numerical results
file = 'Output/Comparison_results.csv'
open(file, 'w')

tprint('diode, cal (uWV^-1), cal error (uWV^-1), V mean (V), V SEM (V), V ptp (V)')

for i in range(0, len(pds)):
    tprint(f'{pds[i]:.0f}, {cal_arr[i]:.1f}, {cal_err[i]:.1f}, {V_avg[i]:#.4g}, {V_sem[i]:#.4g}, {V_ptp[i]:#.4g}')

# colors for plotting
colr = ['royalblue', 'limegreen', 'orange']

# parameters for plotting peak to peak voltage for each photodiode
fig1 = plt.figure(1)
fig1.set_tight_layout(True)

plt.title('Peak-to-peak of DAQ Input Voltage')
plt.xlabel('photodiode no.')
plt.ylabel('ptp voltage $V_{ptp}$ (V)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.bar(pds, V_ptp, color=colr[0])
plt.savefig('Output/Comparison_ptp.png', dpi=300, bbox_inches='tight')

# parameters for plotting SEM of power for each photodiode
fig2 = plt.figure(2)
fig2.set_tight_layout(True)

plt.title('Photodiode Calibration')
plt.xlabel('photodiode no.')
plt.ylabel('cal. constant $C$ ($\mu W V^{-1}$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(pds, cal_arr, cal_err, marker='x', ls='none', capsize=5, c=colr[0])
plt.savefig('Output/Comparison_cal.png', dpi=300, bbox_inches='tight')

# parameters for plotting peak to peak voltage for each photodiode
fig3 = plt.figure(3)
fig3.set_tight_layout(True)

plt.title('Error in Photodiode Calibration')
plt.xlabel('photodiode no.')
plt.ylabel('ptp voltage $V_{ptp}$ (V)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.bar(pds, cal_err, color=colr[0])
plt.ylim(np.amin(cal_err) - 0.2 * np.ptp(cal_err), np.amax(cal_err) + 0.2 * np.ptp(cal_err))
plt.savefig('Output/Comparison_err.png', dpi=300, bbox_inches='tight')

# show the plots
plt.show()    

