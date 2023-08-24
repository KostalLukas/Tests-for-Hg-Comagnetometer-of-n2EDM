# -*- coding: utf-8 -*-
"""
Fiber Test Power Meter Analysis v3.0

Lukas Kostal, 24.8.2023, PSI
"""


import numpy as np
from matplotlib import pyplot as plt


# function to print to console and file simultaneously
def tprint(text=''):
    print(text)
    global file
    with open(file, 'a') as output:
        print(text, file=output)
    

# relative error in power meter measurements
err = 0.05

# specify colors for plotting
colr = ['blue', 'green', 'orange', 'red']

# load the data
data = np.loadtxt('Data/Powermeter_data.csv', unpack=True, delimiter=',', skiprows=1)

# number of measurements
n = len(data[0, :])

# array to hold date in month, day, hour, minute format
date = data[0:4, :]

# specify no of days in each month
# taking 2024 to be a leap year otherwise change 29 -> 28
mtd = np.array([31, 30, 29, 31, 30, 31, 30, 31, 30, 31, 30, 31])

# convert date into elapsed time in h
t = np.zeros(n)
for i in range(0, n):
    t[i] += (date[0, i] - date[0, 0]) * mtd[int(date[0, i] - 1)] * 24
    t[i] += (date[1, i] - date[1, 0]) * 24    
    t[i] += (date[2, i] - date[2, 0])
    t[i] += (date[3, i] - date[3, 0]) / 60

# convert power to mW
data[4:9, :] /= 1000

# slice the data into measurements P in uW and DAQ in V
Ptot = data[4, :] 
Pbst = data[5, :]
Pbsr = data[6, :]
Pfi  = data[7, :]
Pfo  = data[8, :]
Ch1  = data[9, :]
Ch2  = data[10, :]

# calcualte absolute error in uW
Ptot_err = Ptot * err
Pbsr_err = Pbst * err
Pbsr_err = Pbsr * err
Pfi_err = Pfi * err
Pfo_err = Pfo * err

# transmission from power meter measurement with beam splitter
T_pmbs = Pfo / Pbsr / 2
T_pmbs_err = T_pmbs * np.sqrt(2 * err**2)

T_pmbs_avg = np.nanmean(T_pmbs)
T_pmbs_avgerr = np.sqrt(np.nansum(T_pmbs_err**2)) / np.sqrt(np.sum(~np.isnan(T_pmbs_err)))
T_pmbs_ptp = np.ptp(T_pmbs[~np.isnan(T_pmbs)])

# transmission from power meter measurement fiber only
T_pmf = Pfo / Pfi
T_pmf_err = T_pmf * np.sqrt(2 * err**2)

T_pmf_avg = np.nanmean(T_pmf)
T_pmf_avgerr = np.sqrt(np.nansum(T_pmf_err**2)) / np.sqrt(np.sum(~np.isnan(T_pmf_err)))
T_pmf_ptp = np.ptp(T_pmf[~np.isnan(T_pmf)])

cal1_arr = Pbsr / Ch1
cal2_arr = Pfo / Ch2

cal1_err = cal1_arr * err
cal2_err = cal2_arr * err

cal1_avg = np.nanmean(cal1_arr)
cal2_avg = np.nanmean(cal2_arr)

cal1_avgerr = np.sqrt(np.nansum(cal1_err**2)) / np.sqrt(np.sum(~np.isnan(cal1_err)))
cal2_avgerr = np.sqrt(np.nansum(cal2_err**2)) / np.sqrt(np.sum(~np.isnan(cal2_err)))

cal1_ptp = np.ptp(cal1_arr[~np.isnan(cal1_arr)])
cal2_ptp = np.ptp(cal2_arr[~np.isnan(cal2_arr)])

# tranmission from DAQ
T_daq = (Ch2 * cal2_avg) / (Ch1 * cal1_avg) / 2
T_daq_err = T_daq * np.sqrt((cal2_avgerr / cal2_avg)**2 + (cal1_avgerr / cal1_avg)**2)

T_daq_avg = np.nanmean(T_daq)
T_daq_avgerr = np.sqrt(np.nansum(T_daq_err**2)) / np.sqrt(np.sum(~np.isnan(T_daq_err)))
T_daq_ptp = np.ptp(T_daq[~np.isnan(T_daq)])

# beamsplitter ratio of reflected over transmitted
R = Pbsr / Pbst
R_err = R * np.sqrt(2 * err**2)

R_avg = np.nanmean(R)
R_avgerr = np.sqrt(np.nansum(R_err**2)) / np.sqrt(np.sum(~np.isnan(R_err)))
R_ptp = np.ptp(R[~np.isnan(R)])

# print the numerical results
file = 'Output/PM_results.txt'
open(file, 'w')

tprint('Fiber Transmission:')
tprint('PM measurement fiber only:')
tprint(f'T_avg = {T_pmf_avg:.4g}')
tprint(f'T_err = {T_pmf_avg:.4g}')
tprint(f'T_ptp = {T_pmf_avg:.4g}')
tprint('PM measurement:')
tprint(f'T_avg = {T_pmbs_avg:.4g}')
tprint(f'T_err = {T_pmbs_avg:.4g}')
tprint(f'T_ptp = {T_pmbs_avg:.4g}')
tprint('DAQ measurement:')
tprint(f'T_avg = {T_daq_avg:.4g}')
tprint(f'T_err = {T_daq_avg:.4g}')
tprint(f'T_ptp = {T_daq_avg:.4g}')
tprint()
tprint('Beamsplitter ratio:')
tprint('PM measurement:')
tprint(f'R_avg = {R_avg:.4g}')
tprint(f'R_err = {R_avg:.4g}')
tprint(f'R_ptp = {R_avg:.4g}')
tprint('DAQ measurement:')
tprint(f'R_avg = {R_avg:.4g}')
tprint(f'R_err = {R_avg:.4g}')
tprint(f'R_ptp = {R_avg:.4g}')
tprint()
tprint('Calibration constants:')
tprint(f'Ch1_avg = {cal1_avg:.4g} uW')
tprint(f'Ch1_err = {cal1_avgerr:.4g} uW')
tprint(f'Ch1_ptp = {cal1_ptp:.4g} uW')
tprint(f'Ch2_avg = {cal2_avg:.4g} uW')
tprint(f'Ch2_err = {cal2_avgerr:.4g} uW')
tprint(f'Ch2_ptp = {cal2_ptp:.4g} uW')

# parameters for plotting fiber transmission
plt.figure(1)
plt.title('Fiber Transmission over Time', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel(r'transmission $T$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the transmission ratios
plt.errorbar(t, T_pmbs, yerr=T_pmbs_err, fmt='.', capsize=5, c=colr[0], label='$T_{power \; meter}$')
plt.errorbar(t, T_pmf, yerr=T_pmf_err, fmt='.', capsize=5, c=colr[1], label='$T_{fiber \; only}$')
plt.errorbar(t, T_daq, yerr=T_daq_err, fmt='.', capsize=5, c=colr[2], label='$T_{DAQ}$')

plt.legend(loc=(0.1, 1.05), ncol=3)

# save plot
plt.savefig('Output/PM_transmission.png', dpi=300, bbox_inches='tight')

# parameters for plotting beam splitter ratio
plt.figure(2)
plt.title('Beamsplitter Ratio over Time')
plt.xlabel('time $t$ (h)')
plt.ylabel('beamsplitter ratio $R_{R/T}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the beamsplitter ratio
plt.errorbar(t, R, yerr=R_err, fmt='.', capsize=5, c=colr[0])

# save plot
plt.savefig('Output/PM_beamsplitter.png', dpi=300, bbox_inches='tight')
plt.show()

# parameters for plotting total laser output power
plt.figure(3)
plt.title('Total Laser Output Power over Time')
plt.xlabel('time $t$ (h)')
plt.ylabel('output power $P_{out}$ (mW)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the total laser output power
plt.errorbar(t, Ptot, yerr=Ptot_err, fmt='.', capsize=5, c=colr[0])

# save the plot
plt.savefig('Output/PM_power.png', dpi=300, bbox_inches='tight')

# parameters for plotting
plt.figure(4)
plt.title('Photodiode Degradation over Time', pad=40)
plt.xlabel('time $t$ (h)')
plt.ylabel('$P_{DAQ} / P_{power \; meter}$ (unitless)')
plt.rc('grid', linestyle=':', c='black', alpha=0.8)
plt.grid()

# plot the daq calibration constants
plt.errorbar(t, cal1_arr, yerr=cal1_err, fmt='.', capsize=5, c=colr[0], label='DAQ Ch1 (beamsplitter reflection)')
plt.errorbar(t, cal2_arr, yerr=cal2_err, fmt='.', capsize=5, c=colr[3], label='DAQ Ch2 (fiber output)')

plt.legend(loc=(-0.1, 1.05), ncol=2)

# save plot
plt.savefig('Output/PM_calibration.png', dpi=300, bbox_inches='tight')

# show plots
plt.show()