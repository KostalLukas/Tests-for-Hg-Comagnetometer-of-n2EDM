# -*- coding: utf-8 -*-
"""
Fiber Test ADC Analysis v6.0

Lukas Kostal, 5.9.2023, PSI
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


# array of calibration constants and absolute errors in uWV^-1
cal_arr = np.array([2759, 3455, 4900])
cal_err = np.array([138, 173, 200])

# beam splitter ratio and abslute error
Rbs = 0.52
Rbs_err = 0.1

# sampling frequency in Hz
fs = 10

# cutoff frequency in Hz
fc = 'h'

# minimum threshold for fluctuation in power in uW
P_th = 100

# apply a low pass butterworth filter
LPF = True

# subsample data for plotting to save memory
SPLOT = True

# length of the fiber tested
l = 10

# data to be analysed
data = 'UV_new10m_0809_14_42'

# get input parameters
data, SPLOT, P_th, LPF, fc = garg(data, SPLOT, P_th, LPF, fc)

# print update on status
print('reading data')

# load the data
V_arr = np.genfromtxt(f'Data/{data}.txt', skip_footer=1, unpack=True, delimiter=',')

# get the total time of the dataset
t_tot = len(V_arr[0, :]) / fs / 3600

# set all -ve measurements to 0 and discard the first and last 3 measurements
V_arr[V_arr <= 0] = 0
V_arr = V_arr[:, 3:-3]

# get the no of channels in the data either use 2 or 3
n_ch = len(V_arr)
if n_ch > 2:
    n_ch = 3
    V_arr = V_arr[:3, :]

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
    print('calibratng')

# arrays to hold measured power and absolute error in uW
P_arr = np.zeros(V_arr.shape)
P_err = np.zeros(V_arr.shape)

# loop over data for each channel to calibrate and apply LPF
for i in range(0, n_ch):
    # calcualte power from DAQ voltage
    P_arr[i, :] = V_arr[i, :] * cal_arr[i]
    
    # apply low pass butterworth filter
    if LPF == True :   
        P_arr[i, :] = butter_lp(P_arr[i, :], fs, fc, 1)
    P_err[i, :] = P_arr[i, :] * cal_err[i] / cal_arr[i]

# get current no of samples and array of sampling times
n = len(P_arr[0, :])
t = np.arange(0, n) / fs

# print update on status
print('removing measurements below threshold')

# get combined minimum power from Ch1 or Ch2
# also check if calculated fiber transmission will be below 90%
P_min = np.minimum(P_arr[0, :], P_arr[1, :])
fuc = P_arr[1, :] / P_arr[0, :] * Rbs < 0.9
cond = np.logical_or(P_min > P_th, fuc) 

# only take measurements if power is over threshold and fiber transmission below 90%
P_arr = P_arr[:, cond]
P_err = P_err[:, cond]
t_nm = t[~cond] 
t = t[cond]

# update the no of measurements being analysed and get measurement time in h
n = len(t)
th = t / 3600

# print update on status
print('calculating numerical results and errors')
print()

# arrays to hold power fluctuations and associated error in uW
fP_arr = np.zeros(P_arr.shape)
fP_err = np.zeros(P_err.shape)

# arrays to hold rate of power fluctuation and associated error in uWs^-1
dP_arr = np.zeros((n_ch, n-1))
dP_err = np.zeros((n_ch, n-1))

# loop over measurements for each channel to make the calcualtions
for i in range(0, n_ch):
    fP_arr[i, :] = P_arr[i, :] - P_arr[i, 0]
    fP_err[i, :] = np.sqrt(P_err[i, :]**2 + P_err[i, 0]**2)
    
    dP_arr[i, :] = np.diff(P_arr[i, :]) / np.diff(t)
    for j in range(1, n):
        dP_err[i, j-1] = np.sqrt(P_err[i, j-1]**2 + P_err[i, j]**2)

# calculate ratio of power measured by Ch1 and Ch2 and associated error
R21 = P_arr[1, :] / P_arr[0, :]
R21_err = R21 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[0, :] / P_arr[0, :])**2)

# calcuate fiber trasnmission by accounting for beamsplitter ratio and associated error
T = R21 * Rbs
T_err = np.sqrt((R21_err / R21)**2 + (Rbs_err / Rbs)**2)

# calcualte attenuation per m of the fiber and associated error
A = 10 * np.log10(1 / R21) / l
A_err = 10 / np.log(10) / R21 * R21_err / l * 1e-3


# calculate ratio of power measured by Ch2 and Ch3 and associated error
if n_ch > 2:
    R23 = P_arr[1, :] / P_arr[2, :]
    R23_err = R23 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[2, :] / P_arr[2, :])**2)

# calcualte average power and associated error for each channel as well as STD and PTP
P_avg = np.mean(P_arr, axis=1)
P_avg_err = np.sqrt(np.sum(P_err**2, axis=1) / n)
P_std = np.std(P_arr, axis=1)
P_ptp = np.ptp(P_arr, axis=1)

# calcualte average fiber transmission and associated error as well as STD and PTP
T_avg = np.mean(T)
T_avg_err = np.sqrt(np.sum(T_err**2) / n)
T_std = np.std(T)
T_ptp = np.ptp(T)

# calcualte average fiber attenuation per length and associated error
A_avg = np.mean(A)
A_avg_err = np.sqrt(np.sum(A_err**2) / n)

# calcualte RMS rate of change of power for each channel and associated error
dP_rms = np.sqrt(np.sum(dP_arr**2, axis=1) / n)
dP_rms_err = np.sqrt(np.sum(dP_arr**2 * dP_err**2, axis=1) ) / np.sqrt((n-1) * np.sum(dP_arr**2, axis=1))

# rate of change of fiber transmision and associated error
dT = np.diff(R21) / np.diff(t)
dT_err = np.zeros(n-1)
for i in range(1, n):
    dT_err[i-1] = np.sqrt(T_err[i-1]**2 + T_err[i]**2)

# calcaute RMS of rate of change of fiber transmission and associated error
dT_rms = np.sqrt(np.sum(dT**2) / n)
dT_rms_err = np.sqrt(np.sum(dT**2 * dT_err**2) ) / np.sqrt((n-1) * np.sum(dT**2))

# calcaulte average ratio of Ch2 to Ch3 and associated error as well as STD and PTP
if n_ch > 2:
    R23_avg = np.mean(R23)
    R23_avg_err = np.sqrt(np.sum(R23_err**2) / n)
    R23_std = np.std(R23)
    R23_ptp = np.ptp(R23)

# print the numerical results
file = f'Output/DAQ_{data}_results.txt'
open(file, 'w')

tprint(f'dataset:          {data}')
tprint()
tprint(f'total time           = {t_tot:.3f} h')
tprint(f'time above threshold = {th[-1]:.3f} h')
tprint()
tprint(f'Ch1 mean power       = {P_avg[0]:.4g} ± {P_avg_err[0]:.4g} uW')
tprint(f'Ch2 mean power       = {P_avg[1]:.4g} ± {P_avg_err[1]:.4g} uW')
if n_ch > 2:
    tprint(f'Ch3 mean power       = {P_avg[2]:.4g} ± {P_avg_err[2]:.4g} uW')
tprint()
tprint(f'Ch1 ptp power        = {P_ptp[0]:.4g} uW')
tprint(f'Ch2 ptp power        = {P_ptp[1]:.4g} uW')
if n_ch > 2:
    tprint(f'Ch3 ptp power        = {P_ptp[2]:.4g} uW')
tprint()
tprint(f'RMS Ch1 fluctuation  = {dP_rms[0]:4g} uW s^-1')
tprint(f'RMS Ch2 fluctuation  = {dP_rms[1]:4g} uW s^-1')
if n_ch > 2:
    tprint(f'RMS Ch3 fluctuation  = {dP_rms[2]:4g} uW')
tprint()
tprint(f'mean transmission    = {T_avg:.4g} ± {T_avg_err:.4g}')
tprint(f'ptp transmission     = {T_ptp:.4g}')
tprint(f'RMS T fluctuation    = {dT_rms:4g} uW s^-1')
tprint(f'mean attenuation     = {A_avg:.4g} ± {A_avg_err:.4g} dBm^-1')
tprint()
if n_ch > 2:
    tprint(f'mean ratio R23       = {R23_avg:.4g} ± {R23_avg_err:.4g}')
    tprint(f'ptp ratio R23        = {R23_ptp:.4g}')
tprint()
tprint()
tprint(f'Ch1 cal              = {cal_arr[0]} uWV^-1')
tprint(f'Ch2 cal              = {cal_arr[1]} uWV^-1')
tprint(f'Ch3 cal              = {cal_arr[2]} uWV^-1')
tprint(f'threshold            = {P_th:.4g} uW')
tprint(f'sampling freq        = {fs:.4g} Hz')
tprint(f'low pass filter      = {LPF}')
tprint(f'cutoff freq          = {fc:.4g} Hz')
tprint(f'beamsplitter ratio   = {Rbs}')
tprint(f'fiber length         = {l:.1f} m')

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
    dT = dT[::sval]
    dT_err = dT_err[::sval]
    R23 = R23[::sval]
    R23_err = R23_err[::sval]

# colors for plotting
colr = ['royalblue', 'orange', 'limegreen']

# transparency and line width for plotting error regions
alph = 0.35
lw = 1.6

# labels for plotting
labels = ['Ch1 (BS reflection)', 'Ch2 (fiber output)', 'Ch3 (internal FHG)']

# parameters for plotting measured power
plt.figure(1)
plt.title(f'Measured Power over Time \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('power $P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, n_ch):
    plt.plot(th, P_arr[i, :], c=colr[i], label=labels[i], linewidth=lw)
    plt.fill_between(th, P_arr[i, :] - P_err[i, :], P_arr[i, :] + P_err[i, :], \
                     color=colr[i], alpha=alph)
    
plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/DAQ_{data}_P.png', dpi=300, bbox_inches='tight')
        
# parameters for plotting power fluctuation
plt.figure(2)
plt.title(f'Power Fluctuation over Time \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('power fluctuation $\Delta P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, n_ch):
    plt.plot(th, fP_arr[i, :], c=colr[i], label=labels[i], linewidth=lw)
    plt.fill_between(th, fP_arr[i, :] - fP_err[i, :], fP_arr[i, :] + fP_err[i, :], \
                     color=colr[i], alpha=alph)

plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/DAQ_{data}_fP.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of power fluctuation
plt.figure(3)
plt.title(f'Rate of Power Fluctuation over Time \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('rate of power fluctuation $dP/dt$ ($\mu W s^-1$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, n_ch):
    plt.plot(th, dP_arr[i, :], c=colr[i], label=labels[i], linewidth=lw)
    plt.fill_between(th, dP_arr[i, :] - dP_err[i, :], dP_arr[i, :] + dP_err[i, :], \
                     color=colr[i], alpha=alph)
        
plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/DAQ_{data}_dPdt.png', dpi=300, bbox_inches='tight')
        
# parameters for plotting fiber transmission
plt.figure(4)
plt.title(f'Fiber Transmission over Time \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('transmission $T$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, T, c=colr[0])
plt.fill_between(th, T - T_err, T + T_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/DAQ_{data}_T.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of fiber transmission fluctuations
plt.figure(5)
plt.title(f'Rate of Transmission Fluctuation over Time  \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('transmission $dT/dt$ ($s^{-1}$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, dT, c=colr[0])
plt.fill_between(th, dT - dT_err, dT + dT_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/DAQ_{data}_dTdt.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch2 to Ch3
plt.figure(6)
plt.title(f'Ratio of Ch2 to Ch3 \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('ratio $R_{23}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, R23, c=colr[0])
plt.fill_between(th, R23 - R23_err, R23 + R23_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/DAQ_{data}_R23.png', dpi=300, bbox_inches='tight')

plt.show()

