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


# array of range settings on the DAQ
# 2 => ±10V, 3 => ±5V, 4 => ±2.5V, 5 => ±1.25V
ch_set = np.array([5, 5, 5])

# array of calibration constants and absolute errors in uWV^-1
cal_arr = np.array([275.9, 345.5, 490.0])
cal_err = np.array([13.8, 17.3, 20.0])

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
data = 'Diode_082909_D4_D6_Ls'

# get input parameters
data, SPLOT, LPF, fc, FFT= garg(data, SPLOT, LPF, fc, FFT)

# map of DAQ channel voltage ranges in V
ch_map = np.array([10, 5,2.5, 1.25])

# print update on status
print('reading data')

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

R23 = P_arr[1, :] / P_arr[2, :]
R23_err = R23 * np.sqrt((P_err[1, :] / P_arr[1, :])**2 + (P_err[2, :] / P_arr[2, :])**2)

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
file = f'Output/DAQ_{data}_results.txt'
open(file, 'w')

tprint(f'dataset:             {data}')
tprint()
tprint(f'total time           = {th[-1]:.3f} h')
tprint()
tprint(f'Ch1 mean power       = {P_avg[0]:.4g} ± {P_avg_err[0]:.4g} uW')
tprint(f'Ch2 mean power       = {P_avg[1]:.4g} ± {P_avg_err[1]:.4g} uW')
tprint(f'Ch3 mean power       = {P_avg[2]:.4g} ± {P_avg_err[2]:.4g} uW')
tprint()
tprint(f'Ch1 ptp power        = {P_ptp[0]:.4g} uW')
tprint(f'Ch2 ptp power        = {P_ptp[1]:.4g} uW')
tprint(f'Ch3 ptp power        = {P_ptp[2]:.4g} uW')
tprint()
tprint(f'RMS Ch1 fluctuation  = {dP_rms[0]:4g} uW s^-1')
tprint(f'RMS Ch2 fluctuation  = {dP_rms[1]:4g} uW s^-1')
tprint(f'RMS Ch3 fluctuation  = {dP_rms[2]:4g} uW s^-1')
tprint()
tprint(f'R21 mean             = {R21_avg:.4g} ± {R21_avg_err:.4g}')
tprint(f'R21 PTP              = {R21_ptp:.4g}')
tprint(f'R21 RMS fluctuation  = {dR21_rms:4g} uW s^-1')
tprint()
tprint(f'R23 mean             = {R23_avg:.4g} ± {R23_avg_err:.4g}')
tprint(f'R23 PTP              = {R23_ptp:.4g}')
tprint(f'R23 RMS fluctuation  = {dR23_rms:4g} uW s^-1')
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
tprint(f'sampling freq        = {fs:.4g} Hz')
tprint(f'low pass filter      = {LPF}')
tprint(f'cutoff freq          = {fc:.4g} Hz')
tprint(f'FFT Analysis         = {FFT}')

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
    R21 = R21[::sval]
    R21_err = R21_err[::sval]
    R23 = R23[::sval]
    R23_err = R23_err[::sval]
    dR21 = dR21[::sval]
    dR21_err = dR21_err[::sval]
    dR23 = dR23[::sval]
    dR23_err = dR23_err[::sval]

# colors for plotting
colr = ['royalblue', 'orange', 'limegreen']

# transparency and line width for plotting error regions
alph = 0.35
lw = 1.6

# parameters for plotting measured power
plt.figure(1)
plt.title(f'Measured Power over Time \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('power $P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, 3):
    plt.plot(th, P_arr[i, :], c=colr[i], linewidth=lw, label=f'Ch{i+1} - D{pds[i]}')
    plt.fill_between(th, P_arr[i, :] - P_err[i, :], P_arr[i, :] + P_err[i, :], \
                     color=colr[i], alpha=alph)
    
plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_P.png', dpi=300, bbox_inches='tight')
        
# parameters for plotting power fluctuation
plt.figure(2)
plt.title(f'Power Fluctuation over Time \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('power fluctuation $\Delta P$ ($\mu W$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, 3):
    plt.plot(th, fP_arr[i, :], c=colr[i], linewidth=lw, label=f'Ch{i+1} - D{pds[i]}')
    plt.fill_between(th, fP_arr[i, :] - fP_err[i, :], fP_arr[i, :] + fP_err[i, :], \
                     color=colr[i], alpha=alph)

plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/{data}_fP.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of power fluctuation
plt.figure(3)
plt.title(f'Rate of Power Fluctuation over Time \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('rate of power fluctuation $dP/dt$ ($\mu W s^-1$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

for i in range(0, 3):
    plt.plot(th, dP_arr[i, :], c=colr[i], linewidth=lw, label=f'Ch{i+1} - D{pds[i]}')
    plt.fill_between(th, dP_arr[i, :] - dP_err[i, :], dP_arr[i, :] + dP_err[i, :], \
                     color=colr[i], alpha=alph)
        
plt.legend(loc=(0, 1.05), markerscale=20, ncol=3)
plt.savefig(f'Output/DAQ_{data}_dPdt.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch2 to Ch1
plt.figure(4)
plt.title(f'Ratio of Ch2 to Ch1 \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('ratio $R_{21}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, R21, c=colr[0])
plt.fill_between(th, R21 - R21_err, R21 + R21_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/DAQ_{data}_R21.png', dpi=300, bbox_inches='tight')

# parameters for plotting ratio of Ch2 to Ch3
plt.figure(5)
plt.title(f'Ratio of Ch2 to Ch3 \n Dataset: {data}')
plt.xlabel('time $t$ (h)')
plt.ylabel('ratio $R_{23}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, R23, c=colr[0])
plt.fill_between(th, R23 - R23_err, R23 + R23_err, color=colr[0], alpha=alph, linewidth=lw)

plt.savefig(f'Output/DAQ_{data}_R23.png', dpi=300, bbox_inches='tight')

# parameters for plotting rate of change of ratios
plt.figure(6)
plt.title(f'Rate of Change of Ratios \n Dataset: {data}', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('rate of change $dR/dt$ ($s^{-1}$)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

plt.plot(th, dR21, c=colr[0])
plt.fill_between(th, dR21 - dR21_err, dR21 + dR21_err, color=colr[0], \
                 alpha=alph, linewidth=lw, label='ratio $R_{21}$')

plt.plot(th, dR21, c=colr[1])
plt.fill_between(th, dR23 - dR23_err, dR23 + dR23_err, color=colr[1], \
                 alpha=alph, linewidth=lw, label='ratio $R_{23}$')

plt.legend(loc=(0, 1.05), markerscale=20, ncol=2)
plt.savefig(f'Output/DAQ_{data}_dRdt.png', dpi=300, bbox_inches='tight')

if FFT == True:
    # parameters for plotting fft of both ratios
    plt.figure(7)
    plt.title(f'log FFT Spectrum of Ratios \n Dataset: {data}', pad=40)
    plt.xlabel('frequency $f$ (Hz)')
    plt.ylabel('log amplitude $\log_{10}(\mathcal{F}(R))$ (unitless)')
    plt.rc('grid', linestyle=':', c='black', alpha=0.8)
    plt.grid()
    
    plt.plot(fftf, lfftR23, c=colr[1])
    plt.fill_between(fftf, lfftR23 - lfftR23_err, lfftR23 + lfftR23_err, \
                     color=colr[1], alpha=alph, linewidth=lw, label='ratio $R_{23}$')
    
    plt.plot(fftf, lfftR21, c=colr[0])
    plt.fill_between(fftf, lfftR21 - lfftR21_err, lfftR21 + lfftR21_err, \
                     color=colr[0], alpha=alph, linewidth=lw, label='ratio $R_{21}$')
    
    plt.xlim(0, fc/2)
    plt.ylim(np.amin(np.minimum(lfftR21, lfftR23)), np.amax(np.maximum(lfftR21, lfftR23)))
    plt.legend(loc=(0, 1.05), markerscale=20, ncol=2)
    plt.savefig(f'Output/DAQ_{data}_FFT.png', dpi=300, bbox_inches='tight')

# show the plots
plt.show()

# print update on status
print('done')
