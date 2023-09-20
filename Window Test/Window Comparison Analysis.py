# -*- coding: utf-8 -*-
"""
Window Comparison Analysis v1.1

Lukas Kostal, 10.9.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt
import glob as gb
import sys


# ignore all warnings from numpy
np.seterr(all="ignore")


# function to pass arguments from terminal
def parg(*arg_var):
    arg_sys = sys.argv[1:]
    
    arg_name = []
    arg_type = []
    for i in range(0, len(arg_var)):
        arg_id = id(arg_var[i])
        
        for key in globals().keys():
            if key[0] != '_':
                val = globals()[key]
                if id(val) == arg_id:
                    arg_name.append(key)
                    arg_type.append(type(val))
                
    for i in range(0, len(arg_sys)):
        for j in range(0, len(arg_var)):
            if arg_sys[i].split('=')[0] == arg_name[j]:
                
                arg_val = arg_sys[i].split('=')[1]
                
                if arg_val == 'm':
                    arg_val = 1/60
                if arg_val == 'h':
                    arg_val = 1/3600
                
                globals()[arg_name[j]] = arg_type[j](arg_val)
    return None

    
# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)

# sampling frequency in Hz
fs = 10

# no of measurements to average for difference between initial and final transmission
n_ivf = 100
th_ivf = 3

# array of range settings on the DAQ
ch_set = np.array([5, 5, 5])

# array of calibration constants and absolute errors in uWV^-1
cal_arr = np.array([261.3, 353.0, 286600])
cal_err = np.array([18.5, 25.0, 14300])

# map of DAQ channel voltage ranges in V
ch_map = np.array([10, 5, 2.5, 1.25])

# get varables from terminal
parg(n_ivf, th_ivf)

# print update on status
print('loading datasets')

# get all datasets in Data directory and sort according to date
ds = gb.glob('Data/Window_**_W**.txt')
ds.sort()

# lists to hold window number and selected datasets
wn = []
dss = []

# only select datasets the filename of which is 25 characters
for i in range(0, len(ds)):
    if len(ds[i]) == 25:
        wn.append(int(ds[i][-5]))
        dss.append(ds[i])

# change lists to arrays then sort by increasing window number
wn = np.array(wn, dtype=int)
ds = np.array(dss, dtype=str)
sort = np.argsort(wn)
wn = wn[sort]
ds = ds[sort]

# number of windows compared
now = len(ds)

# arrays to hold results for each of the winows
# total exposure time in s
t_ex = np.zeros(now)

# average input power in uW
Pin_avg = np.zeros(now)
Pin_avg_err = np.zeros(now)

# average, peak to peak and difference in intial vs final transmission in uW
T_avg = np.zeros(now)
T_avg_err = np.zeros(now)
T_ptp = np.zeros(now)
T_ivf = np.zeros(now)

# loop over all windows
for i in range(0, now):
    
    # print update on status
    print(f'analysing W{wn[i]} / W{now} \t {ds[i]}')
    
    # load the data
    V_arr = np.genfromtxt(ds[i], skip_footer=1, unpack=True, delimiter=',')

    # set all -ve measurements to 0 and discard the first and last 3 measurements
    V_arr[V_arr <= 0] = 0
    V_arr = V_arr[:, 3:-3]

    # get the no of measurements and measurement times
    n = len(V_arr[0, :])
    t_ex[i] = n / fs
    
    # arrays to hold measured power and absolute error in uW
    P_arr = np.zeros(V_arr.shape)
    P_err = np.zeros(V_arr.shape)

    # loop over data for each channel to calibrate and apply LPF
    for j in range(0, 3):
        # adjust for the DAQ voltage range
        V_arr[j, :] *= ch_map[ch_set[j] - 2]

        # calcualte power from DAQ voltage
        P_arr[j, :] = V_arr[j, :] * cal_arr[j]

    # calculate ratio of power measured by Ch1 and Ch2 and associated error
    R21 = P_arr[1, :] / P_arr[0, :]
    R21_err = R21 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[0, :] / P_arr[0, :])**2)

    # calcuate the window transmission and associated error accounting for beamsplitter
    T = R21 * Rbs[i]
    T_err = np.sqrt((R21_err / R21)**2 + (Rbs_err[i] / Rbs[i])**2)

    T = np.nan_to_num(T, nan=0, posinf=0, neginf=0)
    T_err = np.nan_to_num(T_err, nan=0, posinf=0, neginf=0)

    # calcualte average power and associated error for each channel as well as STD and PTP
    P_avg = np.mean(P_arr, axis=1)
    P_avg_err = np.sqrt(np.sum(P_err**2, axis=1) / n)
    
    # calcualte average window input power and associated error in uW
    Pin_avg[i] = P_avg[0] * Rbs[i]
    Pin_avg_err[i] = Pin_avg[i] * np.sqrt((P_avg_err[0] / P_avg[0])**2 + (Rbs_err[i] / Rbs[i])**2)
    
    # calcualte average of window transmission with error and also STD and PTP
    T_avg[i] = np.mean(T)
    T_avg_err[i] = np.sqrt(np.sum(T_err**2) / n)
    T_ptp[i] = np.ptp(T)
    
    # calcualte difference in initial vs final transmission and write to array
    n_ivf = int(n_ivf)
    N_ivf = int(th_ivf * 3600 * fs)
    T_ivf[i] = np.mean(T[N_ivf:N_ivf+n_ivf]) - np.mean(T[:n_ivf])


# total exposure time in h
th_ex = t_ex / 3600

Pin_all = np.mean(Pin_avg)
Pin_all_err = np.sqrt(np.sum(Pin_avg_err**2) / now)

T_avg_all = np.mean(T_avg[:-1])
T_avg_all_err = np.sqrt(np.sum(T_avg_err[:-1]**2) / (now -1))

# print update on status
print('printing results')
print()

# print the numerical results
file = 'Output/Comparison_results.csv'
open(file, 'w')

tprint('window, Pin (uW), Pin error (uW), T (unitless), T err (unitless), T ivf (unitless)')

for i in range(0, now):
    tprint(f'{wn[i]}, {Pin_avg[i]:.1f}, {Pin_avg_err[i]:.1f}, {T_avg[i]:#.4g}, {T_avg_err[i]:#.4g}, {T_ivf[i]:#.4g}')
            
print()
print(f'mean input power Pin          = {Pin_all:#4g} ± {Pin_all_err:#.4g} uW')
print(f'mean (coated) transmission T  = {T_avg_all:#4g} ± {T_avg_all_err:#.4g}')

# print update on status
print()    
print('plotting')

# colors for plotting
colr = ['royalblue', 'limegreen', 'orange']

# parameters for plotting peak to peak voltage for each photodiode
fig1 = plt.figure(1)
fig1.set_tight_layout(True)

plt.title('Mean Window Input Power')
plt.xlabel('window no.')
plt.ylabel('input power $P_{in}$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(wn, Pin_avg, Pin_avg_err, marker='x', ls='', capsize=5, c=colr[0])

plt.savefig('Output/Comparison_power.png', dpi=300, bbox_inches='tight')

# parameters for plotting peak to peak voltage for each photodiode
fig2 = plt.figure(2)
fig2.set_tight_layout(True)

plt.title('Mean Window Transmission')
plt.xlabel('window no.')
plt.ylabel('transmission $T$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.errorbar(wn, T_avg, T_avg_err, marker='x', ls='', capsize=5, c=colr[0])

plt.savefig('Output/Comparison_transmission.png', dpi=300, bbox_inches='tight')

# parameters for plotting peak to peak voltage for each photodiode
fig3 = plt.figure(3)
fig3.set_tight_layout(True)

plt.title(f'Change in Window Transmission after {th_ivf} h')
plt.xlabel('window no.')
plt.ylabel('change in transmission $\Delta T$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.xticks(wn)

plt.axhline(y=0, c='black')
plt.bar(wn, T_ivf, color=colr[0])

plt.savefig('Output/Comparison_change.png', dpi=300, bbox_inches='tight')

plt.show()

# print update on status
print('done')
