# -*- coding: utf-8 -*-
"""
Window Transmission Test Analysis v2.1

Lukas Kostal, 7.9.2023, PSI
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


# function to get symmetric twin y axis to allign y=0
def set_scales(ax1, ax2):
    y1_min, y1_max = ax1.set_ylim()
    y2_min, y2_max = ax2.set_ylim()
    
    y1_lim = np.maximum(-y1_min, y1_max) * 1.1
    y2_lim = np.maximum(-y2_min, y2_max) * 1.1
    
    ax1.set_ylim(-y1_lim, y1_lim)
    ax2.set_ylim(-y2_lim, y2_lim)
    
    return None


# beam splitter ratio and abslute error
Rbs = 0.525
Rbs_err = 0.037

# sampling frequency in Hz
fs = 10

# cutoff frequency in Hz
fc = 'm'

# minimum threshold for fluctuation in power in uW
P_th = 100

# apply a low pass butterworth filter
LPF = True

# carry out FFT analysis
FFT = True

# subsample data for plotting to save memory
SPLOT = True

# data to be analysed
data = 'Window_082318_W2'

# array of range settings on the DAQ
# 2 => ±10V, 3 => ±5V, 4 => ±2.5V, 5 => ±1.25V
ch_set = np.array([5, 5, 5])

# array of calibration constants and absolute errors in uWV^-1
cal_arr = np.array([262.5, 358.7, 286600])
cal_err = np.array([13.1, 17.9, 14300])

# get input parameters
data, SPLOT, LPF, fc, FFT= garg(data, SPLOT, LPF, fc, FFT)

# map of DAQ channel voltage ranges in V
ch_map = np.array([10, 5, 2.5, 1.25])

# print update on status
print('loading data')

# number of window tested
win = int(data[-1])

# load the data
V_arr = np.genfromtxt(f'Data/{data}.txt', skip_footer=1, unpack=True, delimiter=',')

# set all -ve measurements to 0 and discard the first and last 3 measurements
V_arr[V_arr <= 0] = 0
V_arr = V_arr[:, 3:-3]

# get the no of measurements and measurement times
n = len(V_arr[0, :])
t = np.arange(0, n) / fs
th = t / 3600

# set standard cutoff frequencies with 1min or 1h periods
if LPF == True:
    if type(fc) == str:
        if fc == 'm':
            fc = 1 / 60
        if fc == 'h':
            fc = 1 / 3600
            
    # print update on status  
    print('calibrating and applying LPF')
else:
    fc = fs
    
    print('calibratng')

# arrays to hold measured power and absolute error in uW
P_arr = np.zeros(V_arr.shape)
P_err = np.zeros(V_arr.shape)

# loop over data for each channel to calibrate and apply LPF
for i in range(0, 3):
    # adjust for the DAQ voltage range
    V_arr[i, :] *= ch_map[ch_set[i] - 2]
    
    # calcualte power from DAQ voltage
    P_arr[i, :] = V_arr[i, :] * cal_arr[i]
    
    # apply low pass butterworth filter
    if LPF == True :   
        P_arr[i, :] = butter_lp(P_arr[i, :], fs, fc, 1)
    P_err[i, :] = P_arr[i, :] * cal_err[i] / cal_arr[i]
    
# print update on status
print('analysing') 
    
# arrays to hold power fluctuations and associated error in uW
fP_arr = np.zeros(P_arr.shape)
fP_err = np.zeros(P_err.shape)

# arrays to hold rate of power fluctuation and associated error in uWs^-1
dP_arr = np.zeros((3, n-1))
dP_err = np.zeros((3, n-1))

# loop over measurements for each channel to make the calcualtions
for i in range(0, 3):
    fP_arr[i, :] = P_arr[i, :] - P_arr[i, 0]
    fP_err[i, :] = np.sqrt(P_err[i, :]**2 + P_err[i, 0]**2)
    
    dP_arr[i, :] = np.diff(P_arr[i, :]) / np.diff(t)
    for j in range(1, n):
        dP_err[i, j-1] = np.sqrt(P_err[i, j-1]**2 + P_err[i, j]**2)
        
# calculate ratio of power measured by Ch1 and Ch2 and associated error
R21 = P_arr[1, :] / P_arr[0, :]
R21_err = R21 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[0, :] / P_arr[0, :])**2)

# calcuate the window transmission and associated error accounting for beamsplitter
T = R21 * Rbs
T_err = np.sqrt((R21_err / R21)**2 + (Rbs_err / Rbs)**2)

R23 = P_arr[1, :] / P_arr[2, :]
R23_err = R23 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[2, :] / P_arr[2, :])**2)

# print update on status
print('calculating numerical results and errors')
print()

# calcualte average power and associated error for each channel as well as STD and PTP
P_avg = np.mean(P_arr, axis=1)
P_avg_err = np.sqrt(np.sum(P_err**2, axis=1) / n)
P_std = np.std(P_arr, axis=1)
P_ptp = np.ptp(P_arr, axis=1)

Pin_avg = P_avg[0] * Rbs
Pin_avg_err = Pin_avg * np.sqrt((P_avg[0] / P_avg[0])**2 + (Rbs_err / Rbs)**2)
Pin_std = P_std[0] * Rbs
Pin_ptp = P_ptp[0] * Rbs

# calcualte RMS rate of change of power for each channel and associated error
dP_rms = np.sqrt(np.sum(dP_arr**2, axis=1) / n)
dP_rms_err = np.sqrt(np.sum(dP_arr**2 * dP_err**2, axis=1) ) / np.sqrt((n-1) * np.sum(dP_arr**2, axis=1))

dPin_rms = dP_rms[0] / Rbs
dPin_rms_err = Pin_avg * np.sqrt((dP_rms_err[0] / dP_rms[0])**2 + (Rbs_err / Rbs)**2)

# rate of change of ratio window transmission and associated error
dT = np.diff(T) / np.diff(t)
dT_err = np.zeros(n-1)
for i in range(1, n):
    dT_err[i-1] = np.sqrt(T_err[i-1]**2 + T_err[i]**2)
    
# rate of change of ratio R23 and associated error
dR23 = np.diff(R23) / np.diff(t)
dR23_err = np.zeros(n-1)
for i in range(1, n):
    dR23_err[i-1] = np.sqrt(R23_err[i-1]**2 + R23_err[i]**2)

# calcualte average of window transmission with error and also STD and PTP
T_avg = np.mean(T)
T_avg_err = np.sqrt(np.sum(T_err**2) / n)
T_std = np.std(T)
T_ptp = np.ptp(T)

# calcualte average of the R23 ratio with error and also STD and PTP
R23_avg = np.mean(R23)
R23_avg_err = np.sqrt(np.sum(R23_err**2) / n)
R23_std = np.std(R23)
R23_ptp = np.ptp(R23)

# calcualte RMS of rate of change of window transmission
dT_rms = np.sqrt(np.sum(dT**2) / n)
dT_rms_err = np.sqrt(np.sum(dT**2 * dT_err**2) ) / np.sqrt((n-1) * np.sum(dT**2))

# calcualte RMS of rate of change of ratio R23 and associated error
dR23_rms = np.sqrt(np.sum(dR23**2) / n)
dR23_rms_err = np.sqrt(np.sum(dR23**2 * dR23_err**2) ) / np.sqrt((n-1) * np.sum(dR23**2))

# print update on status
print('printing results')
print()

# print the numerical results
file = f'Output/{data}_results.txt'
open(file, 'w')

tprint(f'dataset:               {data}')
tprint()
tprint(f'total time            = {th[-1]:.3f} h')
tprint()
tprint(f'Pin mean              = {Pin_avg_err:.4g} ± {Pin_avg_err:.4g} uW')
tprint(f'Pin ptp               = {Pin_ptp:.4g} uW')
tprint(f'Pin RMS dP/dt         = {dPin_rms:.4g} ± {dPin_rms_err:.4g} uWs^-1')
tprint()
tprint(f'Pout mean             = {P_avg[1]:.4g} ± {P_avg_err[1]:.4g} uW')
tprint(f'Pout ptp              = {P_ptp[1]:.4g} uW')
tprint(f'Pout RMS dP/dt        = {dP_rms[1]:.4g} ± {dP_rms_err[1]:.4g} uWs^-1')
tprint()
tprint(f'Ch3 mean power       = {P_avg[2]:.4g} ± {P_avg_err[2]:.4g} uW')
tprint(f'Ch3 ptp power        = {P_ptp[2]:.4g} uW')
tprint(f'Ch3 RMS dP/dt        = {dP_rms[2]:.4g} ± {dP_rms_err[2]:.4g} uWs^-1')
tprint()
tprint(f'mean transmission     = {T_avg:.4g} ± {T_avg_err:.4g}')
tprint(f'ptp transmission      = {T_ptp:.4g} ({T_ptp*100:.1f} %)')
tprint()
tprint(f'mean ratio R23        = {R23_avg:.4g} ± {R23_avg:.4g}')
tprint(f'ptp ratio R23         = {R23_ptp:.4g} ({R23_ptp*100:.1f} %)')
tprint()
tprint(f'Ch1 cal               = {cal_arr[0]} ± {cal_err[0]} uWV^-1')
tprint(f'Ch2 cal               = {cal_arr[1]} ± {cal_err[1]} uWV^-1')
tprint(f'Ch3 cal               = {cal_arr[2]} ± {cal_err[2]} uWV^-1')
tprint()
tprint(f'Ch1 range:              {ch_set[0]} => ±{ch_map[ch_set[0]-2]} V')
tprint(f'Ch2 range:              {ch_set[1]} => ±{ch_map[ch_set[1]-2]} V')
tprint(f'Ch3 range:              {ch_set[2]} => ±{ch_map[ch_set[2]-2]} V')
tprint()
tprint(f'sampling freq         = {fs:.4g} Hz')
tprint(f'low pass filter       = {LPF}')
tprint(f'cutoff freq           = {fc:.4g} Hz')
tprint(f'beamsplitter ratio    = {Rbs} ± {Rbs_err}')

# print update on status
print()
print('plotting')

# if subsampling turned on subsample all of the original arrays
if SPLOT == True:
    sval = int(n / 1000)
    th = th[::sval]
    P_arr = P_arr[:, ::sval]
    P_err = P_err[:, ::sval]
    fP_arr = fP_arr[:, ::sval]
    fP_err = fP_err[:, ::sval]
    dP_arr = dP_arr[:, ::sval]
    dP_err = dP_err[:, ::sval]
    T = T[::sval]
    T_err = T_err[::sval]
    R23 = R23[::sval]
    R23_err = R23_err[::sval]
    dT = dT[::sval]
    dT_err = dT_err[::sval]
    dR23 = dR23[::sval]
    dR23_err = dR23_err[::sval]

# colors for plotting
colr = ['royalblue', 'orange', 'limegreen']

# transparency and line width for plotting error regions
alph = 0.2
lw = 1.8

# labels for plotting
labels = ['Ch1 (BS reflection)', 'Ch2 (window output)', 'Ch3 (internal FHG)']

# parameters for plotting measured power
fig1 = plt.figure(1)

ax1 = fig1.add_subplot(111)
ax2 = ax1.twinx()
axs = [ax1, ax1, ax2]

plt.title(f'Measured Power over Time \n Dataset: {data}', pad=40)
ax1.set_xlabel('time $t$ (h)')
ax1.set_ylabel('power $P$ ($\mu W$)')
ax2.set_ylabel('FHG input $P_{FHG}$ ($m W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
ax1.grid()

for i in range(0, 3):
    axs[i].plot(th, P_arr[i, :], c=colr[i], label=labels[i], linewidth=lw)
    axs[i].fill_between(th, P_arr[i, :] - P_err[i, :], P_arr[i, :] + P_err[i, :], \
                     color=colr[i], alpha=alph)

plt.legend(handles= ax1.lines + ax2.lines, loc=(-0.1, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_P.png', dpi=300, bbox_inches='tight')
        
# parameters for plotting power fluctuation
fig2 = plt.figure(2)

ax1 = fig2.add_subplot(111)
ax2 = ax1.twinx()
axs = [ax1, ax1, ax2]

plt.title(f'Power Fluctuation over Time \n Dataset: {data}', pad=40)
ax1.set_xlabel('time $t$ (h)')
ax1.set_ylabel('power fluctuation $\Delta P$ ($\mu W$)')
ax2.set_ylabel('FHG inout fluctuation $\Delta P_{FHG}$ ($m W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
ax1.grid()

for i in range(0, 3):
    axs[i].plot(th, fP_arr[i, :], c=colr[i], label=labels[i], linewidth=lw)
    axs[i].fill_between(th, fP_arr[i, :] - fP_err[i, :], fP_arr[i, :] + fP_err[i, :], \
                     color=colr[i], alpha=alph)

set_scales(ax1, ax2)
plt.legend(handles= ax1.lines + ax2.lines, loc=(-0.1, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_fP.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of power fluctuation
fig3 = plt.figure(3)

ax1 = fig3.add_subplot(111)
ax2 = ax1.twinx()
axs = [ax1, ax1, ax2]

plt.title(f'Rate of Power Fluctuation over Time \n Dataset: {data}', pad=40)
ax1.set_xlabel('time $t$ (h)')
ax1.set_ylabel('rate of power fluctuation $dP/dt$ ($\mu W s^-1$)')
ax2.set_ylabel('FHG input rate of fluctuation $dP_{FHG}/dt$ ($m W s^-1$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
ax1.grid()

for i in range(0, 3):
    axs[i].plot(th, dP_arr[i, :], c=colr[i], label=labels[i], linewidth=lw)
    axs[i].fill_between(th, dP_arr[i, :] - dP_err[i, :], dP_arr[i, :] + dP_err[i, :], \
                     color=colr[i], alpha=alph)     

set_scales(ax1, ax2)   
plt.legend(handles= ax1.lines + ax2.lines, loc=(-0.1, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_dPdt.png', dpi=300, bbox_inches='tight')

# parameters for plotting window transmission
plt.figure(4)
plt.title(f'Window Transmission over Time \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('transmission $T$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, T, c=colr[0])
plt.fill_between(th, T - T_err, T + T_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/{data}_T.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch2 to Ch3
plt.figure(6)
plt.title(f'Ratio of Ch2 to Ch3 \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('ratio $R_{23}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, R23, c=colr[0])
plt.fill_between(th, R23 - R23_err, R23 + R23_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/{data}_R23.png', dpi=300, bbox_inches='tight')

# show plots
plt.show()

# print update on status
print('done')

