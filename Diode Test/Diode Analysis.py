# -*- coding: utf-8 -*-
"""
Diode Test Analysis v3.0

Lukas Kostal, 6.9.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as ss
import scipy.fft as sf
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
            if not(key in arg_name or key[0] == '_'):
                val = globals()[key]
                if id(val) == arg_id:
                    arg_name.append(key)
                    arg_type.append(type(val))

    for i in range(0, len(arg_sys)):
        for j in range(0, len(arg_name)):
            if arg_sys[i].split('=')[0] == arg_name[j]:

                arg_val = arg_sys[i].split('=')[1]

                if arg_val == 'm':
                    arg_val = 1/60
                if arg_val == 'h':
                    arg_val = 1/3600

                if arg_type[j] == bool:
                    arg_val = arg_val == 'True'
                 
                globals()[arg_name[j]] = arg_type[j](arg_val)
    return None


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


# array of range settings on the DAQ
# 2 => ±10V, 3 => ±5V, 4 => ±2.5V, 5 => ±1.25V
ch_set = np.array([5, 5, 5])

# if set to true calibration constants will be automatically loaded from comparison analysis
ACAL = True

# array of calibration constants and absolute errors in uWV^-1
cal_arr = np.array([0, 0, 286600])
cal_err = np.array([0, 0, 14300])

# sampling frequency in Hz
fs = 10

# cutoff frequency in Hz
fc = 1e-2

# apply a low pass butterworth filter
LPF = True

# carry out FFT analysis
FFT = True

# subsample data for plotting to save memory
SPLOT = True

# data to be analysed
data = 'Diode_090118_D4_D6_Ls'

# ratio of amplifier gains Gch2 / Gch1
Rg = 1.332

# get input parameters
parg(data, ACAL, SPLOT, LPF, fc, FFT)

# map of DAQ channel voltage ranges in V
ch_map = np.array([10, 5, 2.5, 1.25])

# print update on status
print('loading data')

# get identifier for the type of measurement
mes = data[-2:]

# get the photodiode number
pds = np.array([int(data[14]), int(data[17]), 0])

# load the data
V_arr = np.genfromtxt(f'Data/{data}.txt', skip_footer=1, unpack=True, delimiter=',')

# set all -ve measurements to 0 and discard the first and last 3 measurements
V_arr[V_arr <= 0] = 0
V_arr = V_arr[:, 3:-3]

# get the no of measurements and measurement times
n = len(V_arr[0, :])
t = np.arange(0, n) / fs
th = t / 3600

# print update on status
if LPF == True:
    print('calibrating and applying LPF')
else:
    print('calibrating')

# load the calibration constants
if ACAL == True:
    cals = np.loadtxt('Output/Comparison_results.csv', usecols=(0, 1, 2),  delimiter=',', skiprows=1, unpack=True)

    cal_arr[0] = cals[1, cals[0, :] == pds[0]]
    cal_err[0] = cals[2, cals[0, :] == pds[0]]
    cal_arr[1] = cals[1, cals[0, :] == pds[1]] * Rg
    cal_err[1] = cals[2, cals[0, :] == pds[1]] * Rg

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

R21 = np.nan_to_num(R21, nan=0, posinf=0, neginf=0)
R21_err = np.nan_to_num(R21_err, nan=0, posinf=0, neginf=0)

R23 = P_arr[1, :] / P_arr[2, :]
R23_err = R23 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[2, :] / P_arr[2, :])**2)

R23 = np.nan_to_num(R23, nan=0, posinf=0, neginf=0)
R23_err = np.nan_to_num(R23_err, nan=0, posinf=0, neginf=0)

# check if hould do FFT
if FFT == True:
    # print update on status
    print('calculating FFT')
    
    # calcualte FFT of the ratio R21 and associated error and then take the log base 10
    fftR21 = np.abs(sf.rfft(R21)[:-1])
    lfftR21 = np.log(fftR21) / np.log(10)
    fftR21_err = np.abs(sf.rfft(R21_err)[:-1])
    lfftR21_err = fftR21_err / fftR21 / np.log(10)
    
    #calcualte FFT of the ratio R23 and associated error and then take the log base 10
    fftR23 = np.abs(sf.rfft(R23)[:-1])
    lfftR23 = np.log(fftR23) / np.log(10)
    fftR23_err = np.abs(sf.rfft(R23_err)[:-1])
    lfftR23_err = fftR23_err / fftR23 / np.log(10)
    
    # calculate frequencies for the FFT of the ratios in Hz
    fftf = sf.fftfreq(n, d=1/fs)[:int(n/2)]

# print update on status
print('calculating numerical results and errors')
print()

# calcualte average power and associated error for each channel as well as STD and PTP
P_avg = np.mean(P_arr, axis=1)
P_avg_err = np.sqrt(np.sum(P_err**2, axis=1) / n)
P_std = np.std(P_arr, axis=1)
P_ptp = np.ptp(P_arr, axis=1)

# calcualte RMS rate of change of power for each channel and associated error
dP_rms = np.sqrt(np.sum(dP_arr**2, axis=1) / n)
dP_rms_err = np.sqrt(np.sum(dP_arr**2 * dP_err**2, axis=1) ) / np.sqrt((n-1) * np.sum(dP_arr**2, axis=1))

# rate of change of ratio R21 and associated error
dR21 = np.diff(R21) / np.diff(t)
dR21_err = np.zeros(n-1)
for i in range(1, n):
    dR21_err[i-1] = np.sqrt(R21_err[i-1]**2 + R21_err[i]**2)
    
# rate of change of ratio R23 and associated error
dR23 = np.diff(R23) / np.diff(t)
dR23_err = np.zeros(n-1)
for i in range(1, n):
    dR23_err[i-1] = np.sqrt(R23_err[i-1]**2 + R23_err[i]**2)

# calcualte average of the R21 ratio with error and also STD and PTP
R21_avg = np.mean(R21)
R21_avg_err = np.sqrt(np.sum(R21_err**2) / n)
R21_std = np.std(R21)
R21_ptp = np.ptp(R21)

# calcualte average of the R23 ratio with error and also STD and PTP
R23_avg = np.mean(R23)
R23_avg_err = np.sqrt(np.sum(R23_err**2) / n)
R23_std = np.std(R23)
R23_ptp = np.ptp(R23)

# calcualte RMS of rate of change of ratio R21 and associated error
dR21_rms = np.sqrt(np.sum(dR21**2) / n)
dR21_rms_err = np.sqrt(np.sum(dR21**2 * dR21_err**2) ) / np.sqrt((n-1) * np.sum(dR21**2))

# calcualte RMS of rate of change of ratio R23 and associated error
dR23_rms = np.sqrt(np.sum(dR23**2) / n)
dR23_rms_err = np.sqrt(np.sum(dR23**2 * dR23_err**2) ) / np.sqrt((n-1) * np.sum(dR23**2))

# print the numerical results
file = f'Output/{data}_results.txt'
open(file, 'w')

tprint(f'dataset:             {data}')
tprint()
tprint(f'total time           = {th[-1]:.3f} h')
tprint()
tprint(f'Ch1 mean power       = {P_avg[0]:#.4g} ± {P_avg_err[0]:#.4g} uW')
tprint(f'Ch2 mean power       = {P_avg[1]:#.4g} ± {P_avg_err[1]:#.4g} uW')
tprint(f'Ch3 mean power       = {P_avg[2]:#.4g} ± {P_avg_err[2]:#.4g} uW')
tprint()
tprint(f'Ch1 ptp power        = {P_ptp[0]:#.4g} uW')
tprint(f'Ch2 ptp power        = {P_ptp[1]:#.4g} uW')
tprint(f'Ch3 ptp power        = {P_ptp[2]:#.4g} uW')
tprint()
tprint(f'Ch1 RMS dP/dt        = {dP_rms[0]:4g} uW s^-1')
tprint(f'Ch2 RMS dP/dt        = {dP_rms[1]:4g} uW s^-1')
tprint(f'Ch3 RMS dP/dt        = {dP_rms[2]:4g} uW s^-1')
tprint()
tprint(f'R21 mean             = {R21_avg:#.4g} ± {R21_avg_err:#.4g}')
tprint(f'R21 ptp              = {R21_ptp:#.4g}')
tprint(f'R21 RMS dR/dt        = {dR21_rms:4g} ± {dR21_rms_err:4g} uW s^-1')
tprint()
tprint(f'R23 mean             = {R23_avg:#.4g} ± {R23_avg_err:#.4g}')
tprint(f'R23 ptp              = {R23_ptp:#.4g}')
tprint(f'R23 RMS dR/dt        = {dR23_rms:4g} ± {dR23_rms_err:#.4g} uW s^-1')
tprint()
tprint()
tprint(f'Ch1 cal              = {cal_arr[0]} ± {cal_err[0]} uWV^-1')
tprint(f'Ch2 cal              = {cal_arr[1]} ± {cal_err[1]} uWV^-1')
tprint(f'Ch3 cal              = {cal_arr[2]} ± {cal_err[2]} uWV^-1')
tprint()
tprint(f'Ch1 range:             {ch_set[0]} => ±{ch_map[ch_set[0]-2]} V')
tprint(f'Ch2 range:             {ch_set[1]} => ±{ch_map[ch_set[1]-2]} V')
tprint(f'Ch3 range:             {ch_set[2]} => ±{ch_map[ch_set[2]-2]} V')
tprint()
tprint(f'sampling freq        = {fs:#.4g} Hz')
tprint(f'low pass filter      = {LPF}')
tprint(f'cutoff freq          = {fc:#.4g} Hz')
tprint(f'FFT Analysis         = {FFT}')

# print update on status
print()
print('plotting')

# array of times for plotting time derivatives
dth = th[1:]

# if subsampling turned on subsample all of the original arrays
if SPLOT == True:
    sval = int(n / 1000)
    th = th[::sval]
    dth = dth[::sval]
    P_arr = P_arr[:, ::sval]
    P_err = P_err[:, ::sval]
    fP_arr = fP_arr[:, ::sval]
    fP_err = fP_err[:, ::sval]
    dP_arr = dP_arr[:, ::sval]
    dP_err = dP_err[:, ::sval]
    R21 = R21[::sval]
    R21_err = R21_err[::sval]
    R23 = R23[::sval]
    R23_err = R23_err[::sval]
    dR21 = dR21[::sval]
    dR21_err = dR21_err[::sval]
    dR23 = dR23[::sval]
    dR23_err = dR23_err[::sval]

# internal FHG power plotted in mW so convert uW to mW
P_arr[2] *= 1e-3
P_err[2] *= 1e-3
fP_arr[2] *= 1e-3
fP_err[2] *= 1e-3
dP_arr[2] *= 1e-3
dP_err[2] *= 1e-3

# colors for plotting
colr = ['royalblue', 'orange', 'limegreen']
lcolr = ['blue', 'orangered', 'green']

# labels for plotting
labels = [f'Ch1 D{pds[0]}', f'Ch2 D{pds[1]}', 'Ch3 FHG input']

# transparency and line width for plotting error regions
alph = 0.3
lw = 1.8

# parameters for plotting measured power
fig1 = plt.figure(1)
fig1.set_tight_layout(True)

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
    axs[i].plot(th, P_arr[i, :], c=lcolr[i], label=labels[i], linewidth=lw)
    axs[i].fill_between(th, P_arr[i, :] - P_err[i, :], P_arr[i, :] + P_err[i, :], \
                     color=colr[i], alpha=alph)

plt.legend(handles= ax1.lines + ax2.lines, loc=(-0.1, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_P.png', dpi=300, bbox_inches='tight')
        
# parameters for plotting power fluctuation
fig2 = plt.figure(2)
fig2.set_tight_layout(True)

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
    axs[i].plot(th, fP_arr[i, :], c=lcolr[i], label=labels[i], linewidth=lw)
    axs[i].fill_between(th, fP_arr[i, :] - fP_err[i, :], fP_arr[i, :] + fP_err[i, :], \
                     color=colr[i], alpha=alph)

set_scales(ax1, ax2)
plt.legend(handles= ax1.lines + ax2.lines, loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_fP.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of power fluctuation
fig3 = plt.figure(3)
fig3.set_tight_layout(True)

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
    axs[i].plot(dth, dP_arr[i, :], c=lcolr[i], label=labels[i], linewidth=lw)
    axs[i].fill_between(dth, dP_arr[i, :] - dP_err[i, :], dP_arr[i, :] + dP_err[i, :], \
                     color=colr[i], alpha=alph)     

set_scales(ax1, ax2)   
plt.legend(handles= ax1.lines + ax2.lines, loc=(0, 1.05), markerscale=20, ncol=3)

plt.savefig(f'Output/{data}_dPdt.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch2 to Ch1
fig4 = plt.figure(4)
fig4.set_tight_layout(True)

plt.title(f'Ratio of Ch2 to Ch1 \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('ratio $R_{21}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, R21, c=lcolr[0])
plt.fill_between(th, R21 - R21_err, R21 + R21_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/{data}_R21.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch2 to Ch3
fig5 = plt.figure(5)
fig5.set_tight_layout(True)

plt.title(f'Ratio of Ch2 to Ch3 \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('ratio $R_{23}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, R23, c=lcolr[0])
plt.fill_between(th, R23 - R23_err, R23 + R23_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/{data}_R23.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of change of ratios
fig6 = plt.figure(6)
fig6.set_tight_layout(True)

ax1 = fig6.add_subplot(111)
ax2 = ax1.twinx()

plt.title(f'Rate of Change of Ratios \n Dataset: {data}', pad=40)
ax1.set_xlabel('time $t$ (h)')
ax1.set_ylabel('rate of change of $R_{21}$ ($s^{-1}$)')
ax2.set_ylabel('rate of change of $R_{23}$ ($s^{-1}$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
ax1.grid()

ax1.plot(dth, dR21, c=lcolr[0], label='ratio $R_{21}$')
ax1.fill_between(dth, dR21 - dR21_err, dR21 + dR21_err, color=colr[0], \
                 alpha=alph, linewidth=lw)

ax2.plot(dth, dR23, c=lcolr[1], label='ratio $R_{23}$')
ax2.fill_between(dth, dR23 - dR23_err, dR23 + dR23_err, color=colr[1], \
                 alpha=alph, linewidth=lw)
    
set_scales(ax1, ax2)  
plt.legend(handles=ax1.lines + ax2.lines, loc=(0, 1.05), ncol=2)
plt.savefig(f'Output/{data}_dRdt.png', dpi=300, bbox_inches='tight')

if FFT == True:
    # parameters for plotting fft of both ratios
    fig7 = plt.figure(7)
    fig7.set_tight_layout(True)
    
    plt.title(f'log FFT Spectrum of Ratios \n Dataset: {data}', pad=40)
    plt.xlabel('frequency $f$ (Hz)')
    plt.ylabel('log amplitude $\log_{10}(\mathcal{F}(R))$ (unitless)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()
    
    plt.plot(fftf, lfftR21, c=lcolr[0], label='ratio $R_{21}$')
    plt.fill_between(fftf, lfftR21 - lfftR21_err, lfftR21 + lfftR21_err, \
                     color=colr[0], alpha=alph, linewidth=lw)
    
    plt.plot(fftf, lfftR23, c=lcolr[1], label='ratio $R_{23}$')
    plt.fill_between(fftf, lfftR23 - lfftR23_err, lfftR23 + lfftR23_err, \
                     color=colr[1], alpha=alph, linewidth=lw)
    
    plt.xlim(0, fc/2)
    plt.legend(loc=(0, 1.05), markerscale=20, ncol=2)
    plt.savefig(f'Output/{data}_FFT.png', dpi=300, bbox_inches='tight')

# show the plots
plt.show()

# print update on status
print('done')
